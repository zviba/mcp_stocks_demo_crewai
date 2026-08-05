# crew.py — the agentic layer, and the MCP *client*.
#
# This is the file the refactor was about. The previous version imported the
# tool functions straight out of mcp_server.py and re-wrapped them with CrewAI's
# @tool decorator, so the agents called Python functions in this process and the
# @mcp.tool() decorators were decoration in the literal sense — no server was
# ever started, no JSON-RPC ever crossed a boundary.
#
# Now the crew is a genuine MCP client: it spawns `python -m stocks_crew.mcp_server`
# as a child process and reaches its tools over real JSON-RPC on stdio, exactly
# the way an external MCP host (Claude Desktop, an IDE, another team's agent)
# would. Nothing in this file knows what the tools *do*; it only knows their
# names, which is the point of the protocol.
from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from .guardrails import check_report
from .trace import ToolTracer

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Roles, backstories and task briefs live in YAML rather than in this file, so a
# prompt can be tweaked without touching Python. See config/agents.yaml and
# config/tasks.yaml.
CONFIG_DIR = Path(__file__).parent / "config"

# Letters, digits, dot, hyphen — ordinary tickers plus the suffixed forms
# yfinance uses (BRK-B, TSM.TW). What it does NOT allow is the point: `symbol`
# is untrusted text that gets interpolated into three agent prompts and passed
# to MCP tools as an argument, so spaces, quotes and punctuation are rejected
# before "AAPL. Ignore previous instructions..." can become instructions.
SYMBOL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,9}$")


def validate_symbol(symbol: str) -> str:
    """Return the normalized upper-case ticker, or raise ValueError."""
    candidate = (symbol or "").strip()
    if not candidate:
        raise ValueError("symbol is required")
    if not SYMBOL_RE.match(candidate):
        raise ValueError(
            "symbol must be 1-10 characters of letters, digits, '.' or '-' "
            "(e.g. AAPL, BRK-B)"
        )
    return candidate.upper()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _load_config(name: str) -> dict:
    """Read and cache config/<name>.yaml."""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class _SafeDict(dict):
    """Leaves unknown {placeholders} alone instead of raising KeyError."""

    def __missing__(self, key):
        return "{" + key + "}"


def _render(value: Any, ctx: dict) -> Any:
    """Fill {placeholders} through nested strings/lists/dicts of a config block."""
    if isinstance(value, str):
        return value.format_map(_SafeDict(ctx)).strip()
    if isinstance(value, list):
        return [_render(v, ctx) for v in value]
    if isinstance(value, dict):
        return {k: _render(v, ctx) for k, v in value.items()}
    return value


def default_agent_tools() -> Dict[str, List[str]]:
    """The per-agent tool allow-lists as declared in agents.yaml."""
    return {
        name: list(spec.get("tools") or [])
        for name, spec in _load_config("agents").items()
    }


# ---------------------------------------------------------------------------
# The MCP connection
# ---------------------------------------------------------------------------


def server_params():
    """Describe how to launch the MCP server as a child process.

    Uses `sys.executable -m` rather than a hardcoded "python3" so the child
    lands in the same (uv-managed) virtualenv as the parent.
    """
    from mcp import StdioServerParameters

    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "stocks_crew.mcp_server"],
        env={**os.environ},
    )


def _tool_description(tool) -> str:
    """The docstring the server sent, without CrewAI's schema preamble.

    The MCP adapter rewrites `description` into a composite that begins with
    "Tool Name: …\nTool Arguments: {…}\nTool Description: <the real one>". For a
    UI listing we only want the last part.
    """
    raw = (getattr(tool, "description", "") or "").strip()
    marker = "Tool Description:"
    return raw.split(marker, 1)[1].strip() if marker in raw else raw


def list_mcp_tools() -> List[Dict[str, str]]:
    """Start the MCP server and ask it what it offers.

    This is a real handshake, not a hardcoded list — which is exactly what makes
    it worth showing in the UI. If someone deletes a @mcp.tool() decorator, this
    stops reporting the tool.
    """
    from crewai_tools import MCPServerAdapter

    with MCPServerAdapter(server_params()) as mcp_tools:
        return [
            {"name": getattr(t, "name", "?"), "description": _tool_description(t)}
            for t in mcp_tools
        ]


def _select_tools(mcp_tools, wanted: set):
    """Pick the named MCP tools out of the live adapter list."""
    if not wanted:
        return []
    selected = [t for t in mcp_tools if getattr(t, "name", "") in wanted]
    missing = wanted - {getattr(t, "name", "") for t in selected}
    if missing:
        logger.warning(
            "Requested MCP tools not offered by the server: %s (server offers %s)",
            sorted(missing),
            sorted(getattr(t, "name", "?") for t in mcp_tools),
        )
    return selected


# ---------------------------------------------------------------------------
# Building the crew
# ---------------------------------------------------------------------------


def build_crew(
    mcp_tools,
    symbol: str,
    language: str = "en",
    tone: str = "professional",
    horizon_days: int = 30,
    tool_overrides: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Any, Dict[str, List[str]]]:
    """Assemble the sequential crew described by config/*.yaml around live MCP tools.

    Returns (crew, tools_per_agent) so the caller can report which tools each
    agent was actually given — the allow-list, before anyone has called anything.
    """
    from crewai import LLM, Agent, Crew, Process, Task

    llm = LLM(model=DEFAULT_MODEL, temperature=0.2)

    ctx = {
        "symbol": symbol,
        "language": language,
        "tone": tone,
        "horizon_days": horizon_days,
    }
    overrides = tool_overrides or {}

    agents: Dict[str, Any] = {}
    granted: Dict[str, List[str]] = {}
    for name, spec in _load_config("agents").items():
        spec = _render(spec, ctx)
        wanted = overrides.get(name, spec.get("tools") or [])
        selected = _select_tools(mcp_tools, set(wanted))
        granted[name] = [getattr(t, "name", "?") for t in selected]
        agents[name] = Agent(
            role=spec["role"],
            goal=spec["goal"],
            backstory=spec["backstory"],
            tools=selected,
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )

    tasks: Dict[str, Any] = {}
    for name, spec in _load_config("tasks").items():
        spec = _render(spec, ctx)
        tasks[name] = Task(
            description=spec["description"],
            expected_output=spec["expected_output"],
            agent=agents[spec["agent"]],
            # Task order in the YAML is the run order, so anything a task lists
            # as context has already been built by the time we get here.
            context=[tasks[c] for c in spec.get("context") or []],
        )

    crew = Crew(
        agents=list(agents.values()),
        tasks=list(tasks.values()),
        process=Process.sequential,
        verbose=True,
    )
    return crew, granted


# ---------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------


def _resolve_api_key(openai_api_key: str = "") -> Optional[str]:
    """Per-request key wins over the environment; returns None if neither is set."""
    key = (openai_api_key or "").strip() or (os.getenv("OPENAI_API_KEY") or "").strip()
    return key or None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _failed(symbol: str, error: str, code: str = "analysis_failed") -> Dict[str, Any]:
    return {
        "success": False,
        "error": error,
        "error_code": code,
        "symbol": symbol,
        "timestamp": _now(),
    }


def run_analysis(
    symbol: str,
    openai_api_key: str = "",
    language: str = "en",
    tone: str = "professional",
    horizon_days: int = 30,
    tool_overrides: Optional[Dict[str, List[str]]] = None,
    progress: Optional[Callable[[str, int], None]] = None,
    on_tool_call: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run the three-agent crew against the MCP server and return a result dict.

    Always returns a dict; failures are reported in-band as
    {"success": False, "error": ..., "error_code": ...} rather than raised, so
    both the HTTP layer and Streamlit can render them the same way.

    On success the dict carries `tool_trace` — the record of every tool call the
    agents actually made, collected off the CrewAI event bus (see trace.py).
    """

    def step(message: str, percent: int) -> None:
        logger.info("[%s] %s", percent, message)
        if progress is not None:
            progress(message, percent)

    try:
        symbol = validate_symbol(symbol)
    except ValueError as e:
        return _failed(symbol, str(e), code="invalid_symbol")

    key = _resolve_api_key(openai_api_key)
    if not key:
        return _failed(
            symbol,
            "OpenAI API key is required to run the crew. Set OPENAI_API_KEY in "
            ".env or enter one in the sidebar.",
            code="openai_api_key_required",
        )

    # CrewAI reaches the model through LiteLLM, which reads the key from the
    # environment rather than from a client object we could pass around.
    previous_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = key

    try:
        from crewai_tools import MCPServerAdapter

        step("Starting the MCP server…", 10)
        # Context-manager form so the MCP subprocess is always torn down, even
        # if the crew raises partway through.
        with MCPServerAdapter(server_params()) as mcp_tools:
            offered = [getattr(t, "name", "?") for t in mcp_tools]
            step(f"MCP server ready — {len(offered)} tools: {', '.join(offered)}", 20)

            crew, granted = build_crew(
                mcp_tools,
                symbol,
                language=language,
                tone=tone,
                horizon_days=horizon_days,
                tool_overrides=tool_overrides,
            )
            step("Crew assembled — running research → technical → report…", 30)

            with ToolTracer(on_call=on_tool_call) as tracer:
                result = crew.kickoff()

        # The output guardrail. The prompts *ask* the agents not to invent
        # figures; this checks the report they actually wrote against the tool
        # output they actually received, in plain Python, after the fact. It
        # reports rather than blocks — see guardrails.check_report.
        report = str(result)
        guardrail = check_report(report, tracer.evidence)
        if not guardrail["passed"]:
            logger.warning(
                "Guardrail failed for %s — ungrounded figures: %s; advice phrases: %s",
                symbol,
                guardrail["ungrounded_figures"],
                [hit["phrase"] for hit in guardrail["advice_phrases"]],
            )

        step("Analysis complete.", 100)
        return {
            "success": True,
            "symbol": symbol,
            "timestamp": _now(),
            "result": report,
            "mcp_tools": offered,
            "agent_tools": granted,
            "guardrail": guardrail,
            **tracer.summary(),
        }
    except Exception as e:
        logger.exception("Crew analysis failed for %s", symbol)
        return _failed(symbol, str(e))
    finally:
        if previous_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = previous_key


if __name__ == "__main__":
    # `uv run python -m stocks_crew.crew --list-tools` — prove the server is real
    # without spending a token.
    from .console import use_utf8_console

    use_utf8_console()
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    if "--list-tools" in sys.argv:
        print(json.dumps(list_mcp_tools(), indent=2))
    else:
        target = next((a for a in sys.argv[1:] if not a.startswith("-")), "AAPL")
        print(json.dumps(run_analysis(target), indent=2))
