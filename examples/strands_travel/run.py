"""Demo 3's looping travel agent, live, with and without LoopFence.

Tools and prompt are adapted from aws-samples/sample-why-agents-fail (MIT-0),
stop-ai-agents-wasting-tokens/03-reasoning-loops-demo.

Runs on a local OpenAI-compatible server (default: Ollama qwen3:14b):

    uv run python examples/strands_travel/run.py --trials 2

Environment overrides: LLM_BASE_URL, LLM_MODEL, LLM_API_KEY.
"""

from __future__ import annotations

import argparse
import logging
import os
import secrets
import time
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)
logging.getLogger("strands").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

from strands import Agent, tool  # noqa: E402
from strands.models.openai import OpenAIModel  # noqa: E402

from loopfence import FenceConfig, LoopFence  # noqa: E402

# ── Ambiguous tools: they never say "done", so the agent keeps retrying ─────────


@tool
def search_flights(origin: str, destination: str, max_price: float = 500.0) -> str:
    """Search for flights between cities under a max price.

    Args:
        origin: Departure city
        destination: Arrival city
        max_price: Maximum price in USD
    """
    prices = [secrets.randbelow(601) + 200 for _ in range(3)]
    airlines = ["AirLine A", "AirLine B", "AirLine C"]
    lines = [f"  {a}: ${p}" for a, p in zip(airlines, prices, strict=True)]
    matching = [p for p in prices if p <= max_price]
    return (
        f"Found {len(matching)} flights under ${max_price:.0f} (out of {len(prices)} checked):\n"
        + "\n".join(lines)
        + "\nNote: More results may be available. Prices change frequently."
    )


@tool
def check_hotel_price(hotel: str, check_in: str) -> str:
    """Check current price for a hotel. Prices fluctuate.

    Args:
        hotel: Hotel name
        check_in: Check-in date
    """
    price = secrets.randbelow(301) + 150
    status = ["limited availability", "selling fast", "few rooms left"][secrets.randbelow(3)]
    return (
        f"{hotel}: ${price}/night ({status}) for {check_in}. "
        "Prices may change — check again for latest."
    )


PROMPT = (
    "You are a persistent travel agent. Always try to find prices within the user's budget. "
    "If results are over budget, search again — prices fluctuate and you might find better "
    "deals on retry. /no_think"
)
TURNS = [
    "Find me the cheapest flight from NYC to Paris under $400 and a hotel under $200/night "
    "for March 15",
    "Try harder, I need it cheaper",
    "Still too expensive. Keep searching until you find something under budget.",
]


def model() -> OpenAIModel:
    return OpenAIModel(
        client_args={
            "base_url": os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
            "api_key": os.getenv("LLM_API_KEY", "ollama"),
        },
        model_id=os.getenv("LLM_MODEL", "qwen3:14b"),
    )


def count_tool_uses(agent: Agent) -> int:
    return sum(
        1
        for msg in agent.messages
        for block in msg.get("content", [])
        if isinstance(block, dict) and "toolUse" in block
    )


def run_trial(fenced: bool, config: FenceConfig, trace_dir: Path, n: int) -> dict[str, Any]:
    fence = LoopFence(config, session_id=f"{'fenced' if fenced else 'open'}-{n}")
    hooks = [fence.strands_hook()] if fenced else []
    agent = Agent(
        model=model(),
        system_prompt=PROMPT,
        tools=[search_flights, check_hotel_price],
        hooks=hooks,
        callback_handler=None,
    )
    start = time.time()
    for i, prompt in enumerate(TURNS, 1):
        result = agent(prompt)
        print(f"    turn {i}: stop_reason={result.stop_reason}", flush=True)
        if fenced and fence.stopped:
            print(f"    fence stopped the run: {fence.report()['stopped_rule']}", flush=True)
            break
    usage = agent.event_loop_metrics.accumulated_usage
    if fenced:
        fence.export_trace(trace_dir / f"trace-{fence.session_id}.json")
    report = fence.report()
    return {
        "mode": "fenced" if fenced else "open",
        "tool_calls_requested": count_tool_uses(agent),
        "tool_calls_run": report["allowed_calls"] if fenced else count_tool_uses(agent),
        "blocked": report["blocked_calls"],
        "stopped_rule": report["stopped_rule"],
        "tokens": int(usage["totalTokens"]),
        "seconds": round(time.time() - start, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=15_000)
    parser.add_argument("--only", choices=["open", "fenced"])
    args = parser.parse_args()

    config = FenceConfig(max_tokens=args.max_tokens)
    trace_dir = Path(__file__).parent / "traces"
    trace_dir.mkdir(exist_ok=True)

    rows: list[dict[str, Any]] = []
    modes = [m == "fenced" for m in ([args.only] if args.only else ["open", "fenced"])]
    for n in range(1, args.trials + 1):
        for fenced in modes:
            print(f"trial {n} [{'fenced' if fenced else 'open'}]", flush=True)
            rows.append(run_trial(fenced, config, trace_dir, n))

    cols = ["mode", "tool_calls_run", "blocked", "stopped_rule", "tokens", "seconds"]
    print("\n" + " | ".join(f"{c:>14}" for c in cols))
    for row in rows:
        print(" | ".join(f"{row[c]!s:>14}" for c in cols))
    print(f"\ntraces: {trace_dir}")


if __name__ == "__main__":
    main()
