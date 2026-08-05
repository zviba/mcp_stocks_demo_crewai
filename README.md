# CrewAI + MCP Stocks Analysis

A teaching demo for **Agents & MCP**. Three CrewAI agents analyse a stock by
calling a **real MCP server** over stdio — and every call they make is recorded,
so you can check the report against what the agents actually looked up.

> **What changed in this version.** The previous version decorated functions with
> `@mcp.tool()` and then imported those same functions directly into `agents.py`,
> re-wrapping them with CrewAI's `@tool`. No server was ever started and no
> JSON-RPC ever crossed a process boundary — the "MCP" was decoration in the
> literal sense. It also depended on `fastmcp`, a different package from the one
> the code imported. Now the crew spawns `python -m stocks_crew.mcp_server` and
> talks to it as a client, exactly the way Claude Desktop or another team's agent
> would. Packaging moved from `requirements.txt` to **uv**.

---

## Quick start

Requires [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then:

```bash
uv sync                     # creates .venv and installs everything from uv.lock
cp .env.sample .env         # add OPENAI_API_KEY or GEMINI_API_KEY (either one)
uv run streamlit run streamlit_crewai_app.py
```

> The first `uv sync` pulls CrewAI, chromadb and the OpenAI and Google SDKs —
> expect a few hundred MB and a minute or two. Subsequent syncs are instant.

That is the whole demo. **There is no API server to start first**: the crew runs
inside the Streamlit process and launches the MCP subprocess itself. The FastAPI
bridge is a separate, optional entry point for HTTP clients:

```bash
uv run uvicorn stocks_crew.api:app --host 0.0.0.0 --port 8001 --reload
```

Prove the MCP server is real before you spend a token:

```bash
uv run python -m stocks_crew.crew --list-tools    # real handshake, prints the tool list
uv run stocks-crew-mcp                            # the server itself; silent stdout is correct
npx @modelcontextprotocol/inspector uv run stocks-crew-mcp
```

---

## Architecture

```
                    ┌────────────────────────┐
   browser ────────▶│ streamlit_crewai_app   │  :8501
                    └───────────┬────────────┘
                                │ in-process call
                    ┌───────────▼────────────┐        ┌──────────────────┐
                    │   stocks_crew.crew     │◀──────▶│ stocks_crew.trace│
                    │   3 agents, sequential │ events │  the tool trace  │
                    └───────────┬────────────┘        └──────────────────┘
                                │ MCPServerAdapter (stdio, JSON-RPC)
                    ┌───────────▼────────────┐
                    │ stocks_crew.mcp_server │  subprocess, FastMCP
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │ tools → analytics      │  pandas + yfinance
                    │        → datasource    │
                    └────────────────────────┘

    optional:  stocks_crew.api  (:8001)  ── data routes call tools/ directly,
                                            /analyze runs the crew over MCP
```

### How the crew talks to the MCP server

`crew.py` spawns the server as a child process and speaks MCP over its
stdin/stdout:

```python
params = StdioServerParameters(
    command=sys.executable,
    args=["-m", "stocks_crew.mcp_server"],
    env={**os.environ},
)
with MCPServerAdapter(params) as mcp_tools:
    ...  # mcp_tools are now CrewAI tools the agents can call
```

Each run spawns a fresh MCP subprocess (~1s of startup). That is deliberate for a
demo: you can watch the process appear and see the tool calls in the logs.

**Consequence:** nothing in the server may write to **stdout**, since stdout *is*
the protocol channel. Use `logging` (which goes to stderr). This is why
`datasource.py` logs its yfinance column debug instead of `print()`ing it.

### The crew

Three agents, run sequentially, all on the same model — `openai/gpt-4o-mini` or
`gemini/gemini-2.5-flash`, depending on which provider the run picked (see
[Choosing a model provider](#choosing-a-model-provider)):

| Agent | MCP tools | Job |
|---|---|---|
| Research Specialist | `search_symbols`, `latest_quote`, `price_series` | Confirm the symbol is real, get the current quote and recent history. |
| Technical Analyst | `indicators`, `detect_events` | Read SMA/EMA/RSI and the gap / volatility / 52-week flags. |
| Report Writer | *none* | Synthesize the two into a report. A writer with tools re-fetches data and quietly contradicts the analyst. |

Roles, backstories and task briefs live in `src/stocks_crew/config/*.yaml`, so a
prompt can be tweaked without touching Python. Task order in `tasks.yaml` *is*
the run order.

### The tool trace — the point of the demo

The task prompts say things like *"Call `latest_quote`. Do not estimate prices."*
That is a **prompt**, not a guarantee. The trace is how you find out whether the
model complied.

`trace.py` subscribes to `crewai_event_bus` and records one entry per tool call:
name, the arguments the model chose, both timestamps, and a preview of the
output. It listens to the framework rather than wrapping the tools, because
after this refactor there is no local function left to wrap — the tools live in
another process behind MCP.

Things worth doing in class:

- Take `latest_quote` away from the Research agent in the sidebar and re-run.
  Watch the report still quote a price, and watch the trace show where it did
  *not* come from.
- Ask for a symbol that Yahoo has no data for. The tools answer
  `{"error": "no_data"}` in band rather than raising, the agent is supposed to
  say so, and the trace shows the call really happened.
- Run with an empty tool list for every agent. The crew produces a confident,
  entirely fabricated report and the trace is empty. That contrast is the lesson.

### There is no `explain` tool

The old server exposed an `explain` tool that called OpenAI from inside the MCP
process. It is gone. The LLM layer sits *above* MCP, in the crew that consumes
these tools — if the crew's own output were also a tool, the crew could call
itself. `tests/test_mcp_server.py` pins this.

### Guardrails

Three of them, at three different points in the run.

**On the input.** `symbol` is untrusted text on its way into three agent prompts
and into MCP tool arguments, so it is validated (`crew.validate_symbol`) before
it can become instructions: letters, digits, `.` and `-`, up to 10 characters. A
bad ticker is a 422 from the API and an `invalid_symbol` result from
`run_analysis` — before any prompt or subprocess runs.

**On the agents' reach.** The `tools:` list in `agents.yaml` is an allow-list per
agent, applied when the crew is built. The report writer has none: a writer with
tools re-fetches data and quietly contradicts the analyst.

**On the output.** `guardrails.py` runs after `crew.kickoff()` and checks the
finished report against the tool output the agents actually received:

- *Grounding* — every figure in the report must match a number some tool
  returned. This is the trace turned from evidence into a verdict.
- *Advice* — the task briefs forbid investment advice, price targets and
  predictions. This checks whether that held.

It is deterministic Python, not a second model, because a guardrail that can
hallucinate is not a guardrail. The result rides back on the `run_analysis`
result as `guardrail` and is rendered between the report and the trace: claim,
verdict, evidence.

Things worth arguing about in class, all of them visible in `guardrails.py`:

- It **reports, it does not block.** Should a failed check withhold the report
  instead, or hand it back to the writer for one retry? Where does that loop end?
- Its tolerances are chosen to under-report. It accepts rounding (`187.7712` →
  `187.77`), rescaling (`44,123,456` → "44.1 million") and a sign carried by the
  prose ("down 1.24" for `-1.24`), and it ignores bare integers under 1000
  because indicator windows are not market figures. Every one of those is a hole
  a determined hallucination could fit through. The alternative is a guardrail
  that fires on every honest run and gets switched off in week two.
- The advice check matches *phrases*, never the bare word "buy" — otherwise the
  mandatory disclaimer ("not a recommendation to buy or sell") fails the report
  that complied. `tests/test_guardrails.py` pins that case.
- It checks against `ToolTracer.evidence`, the full tool output, not the
  240-character `result_preview` the UI shows. A display limit is not a reason to
  blind the check.

Take `latest_quote` away from the Research agent and re-run: the report still
quotes a price, and now the guardrail names the figure that came from nowhere.

---

## Project structure

```
pyproject.toml                deps, entry points, pytest config
uv.lock                       pinned, committed — this is the reproducible bit
streamlit_crewai_app.py       UI (:8501)
src/stocks_crew/
  datasource.py               yfinance + Yahoo search; returns DataFrames/dicts
  analytics.py                pure pandas: SMA, EMA, RSI, gap/vol/52w flags
  tools.py                    the five tool functions; return JSON strings
  mcp_server.py               FastMCP registration + main() -> stdio transport
  crew.py                     the MCP client + the three-agent crew
  trace.py                    records tool calls off the CrewAI event bus
  guardrails.py               output check: report figures vs. tool results
  console.py                  UTF-8 stdout, so CrewAI's emoji do not crash cp1252
  config/agents.yaml          roles, goals, backstories, per-agent tool lists
  config/tasks.yaml           task briefs, run order, context wiring
  api.py                      optional FastAPI bridge (:8001)
tests/                        pytest; no network or LLM calls by default
```

## MCP tools

Served by `stocks-crew-mcp`, all returning JSON strings:

| Tool | Arguments |
|---|---|
| `search_symbols` | `query` |
| `latest_quote` | `symbol` |
| `price_series` | `symbol`, `interval="daily"`, `lookback=180` |
| `indicators` | `symbol`, `window_sma=20`, `window_ema=50`, `window_rsi=14` |
| `detect_events` | `symbol` |

## HTTP API (optional)

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | `{"status": "ok"}` |
| GET | `/mcp/tools` | — | live handshake: `{tools: [{name, description}], count}` |
| GET | `/agents` | — | each agent and the tools it may call |
| POST | `/search` | `{q}` | array of `{symbol, name, region, currency}` |
| POST | `/quote` | `{symbol}` | `{symbol, price, change, change_percent, volume, timestamp}` |
| POST | `/series` | `{symbol, interval, lookback}` | array of OHLCV rows |
| POST | `/indicators` | `{symbol, window_sma, window_ema, window_rsi}` | `{symbol, last_close, sma, ema, rsi}` |
| POST | `/events` | `{symbol}` | `{symbol, date, gap_up, gap_down, vol_spike, is_52w_high, is_52w_low}` |
| GET | `/providers` | — | `{providers: [{name, label, api_key_env, model, configured}], default}` |
| POST | `/analyze` | `{symbol, provider, api_key, language, tone, horizon_days, tools}` | `{success, result, provider, model, tool_trace, agent_tools, mcp_tools, …}` |

The data routes call `tools.py` in-process — a button click should not pay for
spawning a subprocess. `/analyze` and `/mcp/tools` go over MCP, because that is
the part the demo is about.

`tools` on `/analyze` is a per-agent allow-list keyed by agent name from
`agents.yaml`; omitted agents keep their YAML defaults:

```json
{
  "symbol": "AAPL",
  "tools": {"research": ["latest_quote"], "technical": ["indicators", "detect_events"]}
}
```

The older `research_tools` / `technical_tools` / `report_tools` fields still
work and are folded into `tools`.

Tool failures come back as `{"error": "<code>", "message": "..."}` with a 200, so
the UI renders them inline. A malformed *request* is a real 422.

## Choosing a model provider

The crew runs on **OpenAI** or **Google Gemini**. You need a key for one of
them, not both.

```bash
OPENAI_API_KEY=sk-...     # .env — or
GEMINI_API_KEY=AIza...
```

Which one a given run uses is decided in `src/stocks_crew/llm.py`, in this
order:

1. the `provider` named by the caller — the Streamlit sidebar dropdown, or
   `{"provider": "gemini"}` in an `/analyze` body;
2. `LLM_PROVIDER` in `.env`;
3. whichever provider has a key set;
4. OpenAI, which is also whose "key required" error you get when nothing is set.

Everything below that decision is vendor-neutral. CrewAI routes on the *model
string*, written in LiteLLM's `<provider>/<model>` syntax — `openai/gpt-4o-mini`
builds an OpenAI client that reads `OPENAI_API_KEY`, `gemini/gemini-2.5-flash`
builds a Gemini client that reads `GEMINI_API_KEY` — so switching providers is
one model string and one environment variable:

```python
llm = LLM(model="gemini/gemini-2.5-flash", temperature=0.2)
```

`build_crew()` takes that string and never learns which vendor is behind it; the
agents, the MCP server and the guardrail do not change at all. That is the point
worth showing on screen: the provider is a detail of the LLM layer, not of the
agent design.

CrewAI 1.x calls each vendor through its own SDK rather than through LiteLLM, so
`crewai[google-genai]` is in the dependencies — without it, a `gemini/…` model
has nothing to route to. It also validates the model part against a list of
models it knows; a name it does not recognise falls through to LiteLLM, which
this project does not install. If you point `GEMINI_MODEL` or `OPENAI_MODEL` at
something newer than the pinned `crewai`, upgrade `crewai`.

`GET /providers` reports both vendors and which of them currently has a key.

## Environment variables

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Key for the OpenAI provider. Without a key for the chosen provider, `/analyze` returns `openai_api_key_required` / `gemini_api_key_required` — there is no offline fallback. |
| `GEMINI_API_KEY` | Key for the Gemini provider (Google AI Studio). |
| `LLM_PROVIDER` | Optional; `openai` or `gemini`. Which provider a run uses when the caller does not name one. |
| `OPENAI_MODEL` | Optional; defaults to `gpt-4o-mini`. |
| `GEMINI_MODEL` | Optional; defaults to `gemini-2.5-flash`. |

Either model variable takes a bare name or the fully-qualified LiteLLM
`<provider>/<model>` form — the routing prefix is added for you.
| `LOG_LEVEL` | Optional; `DEBUG` shows the MCP handshake and yfinance internals. |

The Streamlit sidebar also accepts a provider and a key at runtime, which take
precedence. Sidebar keys are stored per provider, so switching vendors never
sends an OpenAI key to Google.

## Tests

```bash
uv run pytest              # unit + a real MCP stdio round trip; no network, no tokens
uv run pytest -m network   # additionally hits Yahoo Finance
uv run pytest -m llm       # additionally runs the crew for real (needs a key, spends tokens)
```

`tests/test_mcp_server.py` launches the server as a subprocess and does a real
`list_tools()` handshake — that is the regression test that keeps MCP from
silently becoming decorative again.

## Troubleshooting

- **`openai_api_key_required` / `gemini_api_key_required`** — no key for the
  provider the run chose, neither in `.env` nor typed into the Streamlit
  sidebar. The code tells you which provider it picked and which variable it
  looked for; if that is the wrong provider, set `LLM_PROVIDER` or pass
  `provider` explicitly.
- **The crew hangs, or the adapter reports a protocol error** — almost always
  something printing to stdout inside the server process. Check for stray
  `print()` calls in `datasource.py`, `analytics.py` or `tools.py`.
- **Empty trace after a successful run** — the agents answered without calling
  anything. That is a finding, not a bug; see "The tool trace" above.
- **A run takes a long time** — there is no timeout. The old version had a
  `threading.Timer` that raised in a timer thread, which could not actually
  interrupt `kickoff()`; it was removed rather than left in as reassurance.
  Interrupt the process, or cap `max_iter` on the agents in `crew.build_crew`.
- **Empty series / "No data available"** — Yahoo rate-limits aggressively. Wait a
  minute or try another symbol.
- **`[CrewAIEventsBus] Sync handler error … 'charmap' codec can't encode`
  (Windows)** — CrewAI's console listener prints emoji and the legacy cp1252
  console cannot encode them. Harmless, but noisy in a demo, so the entry points
  call `console.use_utf8_console()`. If you import `stocks_crew.crew` from your
  own script, call it there too.
- **`uv sync` picks the wrong Python** — `.python-version` pins 3.11; run
  `uv python install 3.11` if uv cannot find one.

## License

For demo/educational purposes only. No investment advice.
