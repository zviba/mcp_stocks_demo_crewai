# api.py — HTTP bridge.
#
# Two different paths on purpose:
#   - the data routes call tools.py directly, in process (a button click in the
#     UI should not pay for spawning an MCP subprocess);
#   - /analyze runs the CrewAI crew, which *does* go over MCP, and /mcp/tools
#     does a real handshake so you can see what the server advertises.
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, field_validator

from . import llm as llm_config
from . import tools
from .console import use_utf8_console
from .crew import default_agent_tools, list_mcp_tools, run_analysis, validate_symbol

# Optional: auto-load .env so you don't need --env-file .env
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass

# This module is an application entry point (uvicorn starts here), so setting up
# the console is its job. Without basicConfig the root logger sits at WARNING and
# the crew's "MCP server ready — 5 tools" line stays invisible, which is exactly
# the line you want on screen when demoing this.
use_utf8_console()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(
    title="Stocks CrewAI + MCP Bridge",
    version="1.0.0",
    description=(
        "FastAPI bridge over the yfinance-backed stock tools: search, quote, series, "
        "indicators, events, plus a three-agent CrewAI crew (/analyze) that reaches "
        "those tools over a real MCP stdio connection."
    ),
)

# Wide-open CORS for local dev / Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Utilities
# ---------------------------


def _ok(payload: Any) -> Any:
    """Tools return JSON strings; normalize to Python objects.

    If it is plain text, wrap it under {"text": "..."}.
    """
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except Exception:
            return {"text": payload}
    return payload


def _error(name: str, message: str, status: int = 500, extra: Optional[Dict[str, Any]] = None):
    data: Dict[str, Any] = {"error": name, "message": message}
    if extra:
        data.update(extra)
    return JSONResponse(data, status_code=status)


# ---------------------------
# Schemas (request bodies)
# ---------------------------

# `symbol` is untrusted text on its way into three agent prompts and into MCP
# tool arguments. Validating it here turns a bad ticker into a 422 before the
# handler — and therefore before the prompts and the MCP subprocess — runs.
class _SymbolBody(BaseModel):
    symbol: str = Field(..., min_length=1)

    @field_validator("symbol")
    @classmethod
    def _check_symbol(cls, v: str) -> str:
        return validate_symbol(v)


class SearchBody(BaseModel):
    # Free text on purpose -- this is the one field that is *meant* to be a
    # company name, so it cannot take the symbol guardrail.
    q: str = Field(..., min_length=1, max_length=64, description="Company name or ticker text")


class QuoteBody(_SymbolBody):
    pass


class SeriesBody(_SymbolBody):
    interval: str = Field("daily")
    lookback: int = Field(180, ge=1, le=5000)


class IndicatorsBody(_SymbolBody):
    window_sma: int = Field(20, ge=2, le=500)
    window_ema: int = Field(50, ge=2, le=500)
    window_rsi: int = Field(14, ge=2, le=200)


class EventsBody(_SymbolBody):
    pass


class AnalysisBody(_SymbolBody):
    """Run the CrewAI crew over MCP."""

    provider: Optional[str] = Field(
        None,
        description=(
            "Model vendor: 'openai' or 'gemini'. Omitted, it follows LLM_PROVIDER "
            "or whichever API key is set in the environment."
        ),
    )
    api_key: str = Field(
        "",
        description="Overrides the chosen provider's key (OPENAI_API_KEY / GEMINI_API_KEY)",
    )
    language: str = Field("en", description="Language the report is written in")
    tone: str = Field("professional", description="professional | concise | educational")
    horizon_days: int = Field(30, ge=5, le=365)
    tools: Optional[Dict[str, List[str]]] = Field(
        None,
        description=(
            "Per-agent MCP tool allow-lists, keyed by agent name from agents.yaml "
            '(e.g. {"research": ["latest_quote"], "technical": ["indicators"]}). '
            "Omitted agents keep their YAML defaults."
        ),
    )
    # Legacy fields from earlier versions of this demo, kept so older examples
    # keep working. The tool lists are folded into `tools` below;
    # `openai_api_key` is the single-provider spelling of `api_key`.
    openai_api_key: str = Field("", deprecated=True)
    research_tools: Optional[List[str]] = Field(None, deprecated=True)
    technical_tools: Optional[List[str]] = Field(None, deprecated=True)
    report_tools: Optional[List[str]] = Field(None, deprecated=True)

    def resolved_api_key(self) -> str:
        return (self.api_key or "").strip() or (self.openai_api_key or "").strip()

    def resolved_tool_overrides(self) -> Optional[Dict[str, List[str]]]:
        overrides: Dict[str, List[str]] = dict(self.tools or {})
        for agent, legacy in (
            ("research", self.research_tools),
            ("technical", self.technical_tools),
            ("report", self.report_tools),
        ):
            if legacy is not None:
                overrides.setdefault(agent, legacy)
        return overrides or None


# ---------------------------
# Health
# ---------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------
# Data routes (in-process, no MCP round trip)
# ---------------------------


@app.post("/search")
async def route_search(body: SearchBody):
    try:
        return _ok(tools.search_symbols(body.q))
    except ValidationError as ve:
        return _error("validation_error", ve.json(), status=422)
    except Exception as e:
        return _error("search_failed", str(e), status=500)


@app.post("/quote")
async def route_quote(body: QuoteBody):
    try:
        return _ok(tools.latest_quote(body.symbol))
    except ValidationError as ve:
        return _error("validation_error", ve.json(), status=422)
    except Exception as e:
        return _error("quote_failed", str(e), status=500)


@app.post("/series")
async def route_series(body: SeriesBody):
    try:
        return _ok(tools.price_series(body.symbol, body.interval, body.lookback))
    except ValidationError as ve:
        return _error("validation_error", ve.json(), status=422)
    except Exception as e:
        return _error("series_failed", str(e), status=500)


@app.post("/indicators")
async def route_indicators(body: IndicatorsBody):
    try:
        return _ok(tools.indicators(body.symbol, body.window_sma, body.window_ema, body.window_rsi))
    except ValidationError as ve:
        return _error("validation_error", ve.json(), status=422)
    except Exception as e:
        return _error("indicators_failed", str(e), status=500)


@app.post("/events")
async def route_events(body: EventsBody):
    try:
        return _ok(tools.detect_events(body.symbol))
    except ValidationError as ve:
        return _error("validation_error", ve.json(), status=422)
    except Exception as e:
        return _error("events_failed", str(e), status=500)


# ---------------------------
# MCP + agent routes
# ---------------------------


@app.get("/mcp/tools")
async def route_mcp_tools():
    """Start the MCP server, do the handshake, report what it advertises.

    Note this is not a hardcoded list: it spawns the server and asks. That is
    what makes it a useful check that MCP is really wired up.
    """
    try:
        # MCPServerAdapter drives its own event loop for the stdio subprocess,
        # so it must not run inside this one.
        offered = await asyncio.to_thread(list_mcp_tools)
        return {"tools": offered, "count": len(offered)}
    except Exception as e:
        return _error("mcp_handshake_failed", str(e), status=500)


@app.get("/agents")
async def route_agents():
    """The crew's agents and the MCP tools each one is allowed to call."""
    return {"agents": default_agent_tools()}


@app.get("/providers")
async def route_providers():
    """Which model vendors this build knows, and which have a key right now.

    `configured` is read from the environment at request time, so it answers
    "can /analyze actually run?" without spending a token to find out.
    """
    return {
        "providers": [
            {
                "name": p.name,
                "label": p.label,
                "api_key_env": p.api_key_env,
                "model": llm_config.model_for(p.name),
                "configured": p.name in llm_config.configured_providers(),
            }
            for p in llm_config.PROVIDERS.values()
        ],
        "default": llm_config.default_provider(),
    }


@app.post("/analyze")
async def route_analyze(body: AnalysisBody):
    """Run the three-agent crew and return the report plus the tool-call trace.

    Example:
        {
          "symbol": "AAPL",
          "provider": "gemini",
          "api_key": "AIza...",
          "tools": {"research": ["latest_quote"], "technical": ["indicators", "detect_events"]}
        }
    """
    if body.provider:
        try:
            llm_config.get_provider(body.provider)
        except llm_config.UnknownProvider as e:
            return _error(llm_config.UnknownProvider.error_code, str(e), status=422)

    overrides = body.resolved_tool_overrides()
    known_agents = set(default_agent_tools())
    unknown = sorted(set(overrides or {}) - known_agents)
    if unknown:
        return _error(
            "unknown_agent",
            f"No such agent(s): {unknown}. Known agents: {sorted(known_agents)}",
            status=422,
        )

    try:
        # crew.kickoff() is blocking and the MCP adapter runs its own loop, so
        # this has to leave the request's event loop.
        result = await asyncio.to_thread(
            run_analysis,
            body.symbol,
            provider=body.provider,
            api_key=body.resolved_api_key(),
            language=body.language,
            tone=body.tone,
            horizon_days=body.horizon_days,
            tool_overrides=overrides,
        )
        return _ok(result)
    except ValidationError as ve:
        return _error("validation_error", ve.json(), status=422)
    except Exception as e:
        return _error("analysis_failed", str(e), status=500)
