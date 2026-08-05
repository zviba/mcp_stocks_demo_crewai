# trace.py — "did the agents actually call the tools, or did they make it up?"
#
# That question is the whole point of this demo, so the answer has to come from
# somewhere the model cannot influence. The old version answered it by wrapping
# each tool in a logging decorator — which worked only because the tools were
# being imported and called in-process. Now the tools live behind MCP, in
# another process, and are handed to the crew by MCPServerAdapter: there is no
# local function left to decorate.
#
# So we listen instead. CrewAI emits an event for every tool invocation on
# `crewai_event_bus`, and ToolUsageFinishedEvent carries the tool name, the
# arguments the model chose, both timestamps and the output. Subscribing to that
# gives a trace that covers *every* tool an agent touches, without the tracing
# code having to know anything about MCP.
#
# Handlers run on the bus's thread pool, so `_calls` is guarded by a lock and
# `__exit__` flushes the bus before the trace is read.
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

#: How much of a tool's output to keep in the trace. Enough to prove the value
#: the agent quoted came from the tool; short enough to render in a UI.
PREVIEW_CHARS = 240


def _preview(value: Any) -> str:
    text = value if isinstance(value, str) else str(value)
    return text[:PREVIEW_CHARS]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ToolTracer:
    """Collect one record per tool call made inside the `with` block.

    Usage:

        with ToolTracer() as tracer:
            crew.kickoff()
        tracer.calls  # -> list of dicts

    Pass `on_call` to be notified as each call lands. Note the callback runs on
    the event bus's worker thread, not the caller's — fine for logging or an
    append to a list, not for anything that expects the main thread (Streamlit
    widgets, for one).
    """

    def __init__(self, on_call: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        self._lock = threading.Lock()
        self._calls: List[Dict[str, Any]] = []
        self._evidence: List[str] = []
        self._on_call = on_call
        self._subscriptions: List[tuple] = []

    # -- reading -----------------------------------------------------------

    @property
    def calls(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._calls)

    @property
    def evidence(self) -> List[str]:
        """Every tool output in full, untruncated — what guardrails.py checks against.

        Kept apart from `calls` on purpose. `result_preview` is cut to
        PREVIEW_CHARS because the trace is rendered in a UI and downloaded as
        JSON; a price series would otherwise be thousands of characters. But a
        grounding check run against a truncated preview would flag row 40 of that
        series as an invented number, so the check gets the whole thing. The
        lesson is general: a guardrail is only as good as the evidence you let it
        see, and a display limit is not a reason to blind it.
        """
        with self._lock:
            return list(self._evidence)

    @property
    def successful(self) -> int:
        return sum(1 for c in self.calls if c.get("success"))

    def summary(self) -> Dict[str, Any]:
        calls = self.calls
        return {
            "tool_trace": calls,
            "tool_calls_count": len(calls),
            "tool_calls_successful": sum(1 for c in calls if c.get("success")),
        }

    # -- recording ---------------------------------------------------------

    def _record(self, entry: Dict[str, Any], output: Any = None) -> None:
        with self._lock:
            self._calls.append(entry)
            if output:
                self._evidence.append(output if isinstance(output, str) else str(output))
        if self._on_call is not None:
            try:
                self._on_call(entry)
            except Exception:  # a broken UI callback must not break the crew
                logger.exception("Tool-trace callback raised; continuing.")

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "ToolTracer":
        # Imported here rather than at module scope: importing crewai costs a
        # couple of seconds, and api.py's data routes never need it.
        from crewai.events import (
            ToolUsageErrorEvent,
            ToolUsageFinishedEvent,
            crewai_event_bus,
        )

        def on_finished(source: Any, event: Any) -> None:
            failure = getattr(event, "failure", None)
            finished_at = getattr(event, "finished_at", None)
            output = getattr(event, "output", "")
            try:
                duration = (event.finished_at - event.started_at).total_seconds()
            except Exception:
                duration = 0.0
            self._record(
                {
                    "timestamp": finished_at.isoformat() if finished_at is not None else _now(),
                    "tool_name": event.tool_name,
                    "agent_role": getattr(event, "agent_role", None),
                    "arguments": event.tool_args,
                    "duration_seconds": round(duration, 3),
                    "from_cache": bool(getattr(event, "from_cache", False)),
                    "success": failure is None,
                    "result_preview": _preview(output),
                    "error": str(failure) if failure is not None else None,
                },
                # Only a successful call is evidence. A failed one returns an
                # error string, and its numbers must not ground anything.
                output=output if failure is None else None,
            )

        def on_error(source: Any, event: Any) -> None:
            self._record({
                "timestamp": _now(),
                "tool_name": getattr(event, "tool_name", "?"),
                "agent_role": getattr(event, "agent_role", None),
                "arguments": getattr(event, "tool_args", {}),
                "duration_seconds": 0.0,
                "from_cache": False,
                "success": False,
                "result_preview": "",
                "error": str(getattr(event, "error", "unknown error")),
            })

        for event_type, handler in (
            (ToolUsageFinishedEvent, on_finished),
            (ToolUsageErrorEvent, on_error),
        ):
            crewai_event_bus.on(event_type)(handler)
            self._subscriptions.append((event_type, handler))

        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        from crewai.events import crewai_event_bus

        # Handlers are dispatched to a thread pool, so an event emitted in the
        # last moments of kickoff() may still be in flight. Flush before the
        # caller reads .calls, otherwise the final tool call goes missing.
        try:
            crewai_event_bus.flush(timeout=10)
        except Exception:
            logger.debug("Event bus flush failed; trace may be incomplete.", exc_info=True)

        for event_type, handler in self._subscriptions:
            try:
                crewai_event_bus.off(event_type, handler)
            except Exception:
                logger.debug("Could not unsubscribe %s", event_type, exc_info=True)
        self._subscriptions.clear()
        return False  # never swallow the crew's exception
