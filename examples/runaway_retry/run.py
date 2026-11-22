"""A runaway retry loop inside a single task, with and without LoopFence.

An order-status tool keeps failing with a transient-looking 503, and the agent is told
to retry until it gets the live status. Unfenced, nothing stops it except the demo's
own safety ceiling; fenced, the third identical call is blocked, and repeated blocks
escalate to a hard stop.

Runs on a local OpenAI-compatible server (default: Ollama qwen3:14b):

    uv run python examples/runaway_retry/run.py --trials 2

Environment overrides: LLM_BASE_URL, LLM_MODEL, LLM_API_KEY.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import uuid
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)
logging.getLogger("strands").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

from strands import Agent, tool  # noqa: E402
from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry  # noqa: E402
from strands.models.openai import OpenAIModel  # noqa: E402

from loopfence import FenceConfig, LoopFence  # noqa: E402

SAFETY_CEILING = 25  # demo-only cap so the unfenced run cannot spin forever


@tool
def get_order_status(order_id: str) -> str:
    """Get the live shipping status of a customer order.

    Args:
        order_id: The order identifier, e.g. "A-1042"
    """
    now = datetime.now(UTC).isoformat(timespec="seconds")
    return (
        f"ERROR 503 Service Unavailable: inventory service timed out "
        f"(request_id={uuid.uuid4()}, at {now}). The request may succeed if retried."
    )


PROMPT = (
    "You are an order-support agent. Customers must always receive the live status of their "
    "order. If a tool call fails, retry it: the service usually recovers within a few attempts. "
    "Do not answer the customer until you have the live status. /no_think"
)
QUESTION = "Hi, what's the status of my order A-1042?"


class SafetyCeiling(HookProvider):
    """Stops the *unfenced* demo run after SAFETY_CEILING tool calls."""

    def __init__(self) -> None:
        self.calls = 0

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self._count)

    def _count(self, event: BeforeToolCallEvent) -> None:
        self.calls += 1
        if self.calls > SAFETY_CEILING:
            event.cancel_tool = "Demo safety ceiling reached."
            event.agent.cancel(message="[demo safety ceiling reached]", after_current_tools=True)


def model() -> OpenAIModel:
    return OpenAIModel(
        client_args={
            "base_url": os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
            "api_key": os.getenv("LLM_API_KEY", "ollama"),
        },
        model_id=os.getenv("LLM_MODEL", "qwen3:14b"),
    )


def run_trial(fenced: bool, config: FenceConfig, trace_dir: Path, n: int) -> dict[str, Any]:
    fence = LoopFence(config, session_id=f"{'fenced' if fenced else 'open'}-{n}")
    ceiling = SafetyCeiling()
    hooks: list[HookProvider] = [ceiling]
    if fenced:
        hooks.append(fence.strands_hook())
    agent = Agent(
        model=model(),
        system_prompt=PROMPT,
        tools=[get_order_status],
        hooks=hooks,
        callback_handler=None,
    )
    start = time.time()
    result = agent(QUESTION)
    elapsed = time.time() - start
    usage = agent.event_loop_metrics.accumulated_usage
    report = fence.report()
    if fenced:
        fence.export_trace(trace_dir / f"trace-{fence.session_id}.json")
    executed = report["allowed_calls"] if fenced else min(ceiling.calls, SAFETY_CEILING)
    if fenced:
        ended_by = report["stopped_rule"] or "agent"
    else:
        ended_by = "safety ceiling" if ceiling.calls > SAFETY_CEILING else "agent"
    return {
        "mode": "fenced" if fenced else "open",
        "tool_calls_run": executed,
        "blocked": report["blocked_calls"],
        "ended_by": ended_by,
        "tokens": int(usage["totalTokens"]),
        "seconds": round(elapsed, 1),
        "final_answer": str(result).strip().replace("\n", " ")[:160],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--only", choices=["open", "fenced"])
    args = parser.parse_args()

    trace_dir = Path(__file__).parent / "traces"
    trace_dir.mkdir(exist_ok=True)
    modes = [m == "fenced" for m in ([args.only] if args.only else ["open", "fenced"])]

    rows: list[dict[str, Any]] = []
    for n in range(1, args.trials + 1):
        for fenced in modes:
            print(f"trial {n} [{'fenced' if fenced else 'open'}]", flush=True)
            rows.append(run_trial(fenced, FenceConfig(), trace_dir, n))
            print(f"    answer: {rows[-1]['final_answer']}", flush=True)

    cols = ["mode", "tool_calls_run", "blocked", "ended_by", "tokens", "seconds"]
    print("\n" + " | ".join(f"{c:>14}" for c in cols))
    for row in rows:
        print(" | ".join(f"{row[c]!s:>14}" for c in cols))
    print(f"\ntraces: {trace_dir}")


if __name__ == "__main__":
    main()
