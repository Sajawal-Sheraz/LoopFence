# LoopFence

Stops an AI-agent run from looping or overspending, and records why it stopped.

LLM agents treat a failed or ambiguous tool result as a reason to try again. Without a hard
guard they repeat the same call, invent new arguments, and resend a growing context on every
retry. In our reproduction (a travel agent on `qwen3:14b`, three user turns) a 2-call task
became 8–11 tool calls, and token use grew about 5× across the turns.

LoopFence sits between the agent and its tools. It is a small, dependency-free Python library:
synchronous, in-memory, one fence per agent run.

## Quickstart

```python
from loopfence import FenceConfig, LoopFence

fence = LoopFence(FenceConfig(max_tokens=20_000))


@fence.protect
def search_flights(origin: str, destination: str, max_price: float) -> str: ...


@fence.protect(pollable=True)  # may repeat while its result keeps changing
def get_job_status(job_id: int) -> str: ...


fence.add_tokens(input_tokens, output_tokens)  # after each LLM call
print(fence.report())
fence.export_trace("trace.json")
```

A refused call raises `ToolCallBlocked` (soft: the agent should read the message and continue)
or a `RunStoppedError` subclass, `LoopDetectedError` or `BudgetExceededError` (hard: end the run).

### Strands Agents

```python
agent = Agent(model=..., tools=[...], hooks=[fence.strands_hook(pollable_tools={"get_job_status"})])
```

Soft blocks become the tool's error result, a stop cancels the agent after the current tools,
and token usage is read from the agent automatically. Install with `loopfence[strands]`.

### Any other framework

```python
decision = fence.check(tool_name, args, pollable=False)
if not decision.allowed:
    return decision.message  # or fence.raise_for(decision)
result = run_tool(...)
fence.record_result(decision, result)
```

## Rules

Evaluated in this order for every tool call. Counters are per session (one `LoopFence`) and are
never reset between user turns.

| Rule | Action | Default | What it catches |
|------|--------|---------|-----------------|
| `timeout` | stop | off (`max_seconds`) | slow loops that run for hours |
| `budget` | stop | off (`max_tokens`, `max_usd`) | any loop; the backstop |
| `max_calls` | stop | after 50 tool calls | runaway sessions |
| `exact_repeat` | block | 3rd identical call | retrying the same call |
| `poll_max` | block | 21st identical call to a pollable tool | endless polling |
| `poll_unchanged` | block | after 5 identical results in a row (pollable) | stuck polling |
| `tool_cap` | block | 9th call to one tool, any args (pollable tools exempt unless set in `tool_caps`) | invented-argument loops (partly) |

The 3rd block in a session (`max_blocks`) becomes a stop. Once stopped, every later tool or LLM
check is refused. Block messages tell the agent to reuse earlier results and **not invent data**.
In the original demo, a blocked call led the model to make up a price.

"Identical" means the same tool name and normalized arguments: keys sorted, volatile keys such as
`request_id` and `timestamp` dropped, `400.0` treated as `400`. Results of pollable tools also have
timestamps, UUIDs and long hex IDs masked, so a stuck job whose output only changes its clock
counts as unchanged. A normal tool's repeat is blocked even if its result differs, because random
or fluctuating output is noise, not progress.

## Trace

`fence.export_trace()` returns versioned JSON with the config, a summary, and one event per
decision: `tool_call` (allow/block/stop plus the rule), `tool_result` (hash), `llm_call`, and
`usage`. With an injected clock (`LoopFence(clock=...)`) the trace is byte-identical across runs.

## Live demos

Both demos run against a local Ollama model (`qwen3:14b` by default), with and without the fence.
Set `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY` to use another OpenAI-compatible server.

**Runaway retry** (`examples/runaway_retry/`): an order-status tool always returns a 503, and
the agent is told to retry until it succeeds.

```bash
uv run python examples/runaway_retry/run.py --trials 2
```

| Run | Tool calls | Tokens |
|-----|-----------:|-------:|
| unfenced | 6–11 | 5,865–13,851 |
| fenced | 2 (3rd blocked by `exact_repeat`) | 2,736–3,112 |

After a single block the agent stopped retrying and told the customer the service was down,
without inventing a status. These numbers come from two trials on one model, so treat them as an
illustration, not a benchmark.

**Travel agent** (`examples/strands_travel/`): demo 3's ambiguous travel tools over three pushy
user turns.

```bash
uv run python examples/strands_travel/run.py --trials 2
```

A block saves the tool execution, not the LLM call that requested it. Tokens are saved when the
agent gives up after a block, or when a stop rule (budget, timeout, `max_blocks`) ends the run.

## Known limitations

- Invented-argument loops (a new made-up hotel name every call) are caught only by `tool_cap` or
  the budget. They are not recognised as loops.
- Paraphrased repeats (same meaning, different text) are not detected.
- State lives in one process, so restarts and multiple workers each count separately.
- Tokens are counted after each LLM call, so one large call can overshoot the budget.

## Development

```bash
uv sync --dev
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

Tests never call an LLM. Recorded agent runs in `tests/fixtures/` are replayed offline.
