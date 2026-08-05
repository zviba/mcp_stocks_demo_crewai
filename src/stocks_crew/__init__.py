"""stocks_crew -- a CrewAI crew that talks to a real MCP server.

Layers, bottom up:

    datasource.py   yfinance + Yahoo search; returns DataFrames/dicts
    analytics.py    pure pandas: SMA, EMA, RSI, gap/vol/52w flags
    tools.py        the five tool functions; return JSON strings
    mcp_server.py   FastMCP registration + main() -> stdio transport
    crew.py         the client: spawns mcp_server as a subprocess, hands its
                    tools to three CrewAI agents
    trace.py        records every tool call off the CrewAI event bus
    api.py          FastAPI bridge (:8001)
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
