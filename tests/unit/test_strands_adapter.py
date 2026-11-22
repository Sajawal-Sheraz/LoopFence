"""Strands adapter, driven with real Strands event objects and a stub agent (no LLM)."""

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("strands")

from strands.hooks import (
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
)

from loopfence import FenceConfig, LoopFence


class StubAgent:
    def __init__(self) -> None:
        self.cancelled: list[str | None] = []
        self.event_loop_metrics = SimpleNamespace(
            accumulated_usage={"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
        )

    def cancel(self, *, message: str | None = None, after_current_tools: bool = False) -> None:
        assert after_current_tools
        self.cancelled.append(message)


def tool_call(hook: Any, agent: StubAgent, n: int, name: str, args: dict[str, Any]) -> Any:
    tool_use = {"toolUseId": f"id-{n}", "name": name, "input": args}
    before = BeforeToolCallEvent(
        agent=agent,  # type: ignore[arg-type]
        selected_tool=None,
        tool_use=tool_use,  # type: ignore[arg-type]
        invocation_state={},
    )
    hook._before_tool(before)
    if not before.cancel_tool:
        result = {"toolUseId": f"id-{n}", "status": "success", "content": [{"text": "pending"}]}
        hook._after_tool(
            AfterToolCallEvent(
                agent=agent,  # type: ignore[arg-type]
                selected_tool=None,
                tool_use=tool_use,  # type: ignore[arg-type]
                invocation_state={},
                result=result,  # type: ignore[arg-type]
            )
        )
    return before


def test_soft_block_then_hard_stop() -> None:
    fence = LoopFence()
    hook = fence.strands_hook()
    agent = StubAgent()
    args = {"origin": "NYC", "destination": "Paris", "max_price": 400}
    events = [tool_call(hook, agent, i, "search_flights", args) for i in range(5)]
    assert [bool(e.cancel_tool) for e in events] == [False, False, True, True, True]
    assert "Reuse the earlier result" in events[2].cancel_tool
    assert len(agent.cancelled) == 1  # only the stop cancels the agent
    assert fence.report()["stopped_rule"] == "exact_repeat"


def test_tool_use_id_does_not_break_poll_unchanged() -> None:
    fence = LoopFence(FenceConfig(max_unchanged_results=3))
    hook = fence.strands_hook(pollable_tools={"get_status"})
    agent = StubAgent()
    events = [tool_call(hook, agent, i, "get_status", {"job": 1}) for i in range(4)]
    assert [bool(e.cancel_tool) for e in events] == [False, False, False, True]
    assert fence.report()["blocks_by_rule"] == {"poll_unchanged": 1}


def after_model(usage: dict[str, int] | None) -> AfterModelCallEvent:
    message: dict[str, Any] = {"role": "assistant", "content": [{"text": "ok"}]}
    if usage is not None:
        message["metadata"] = {"usage": usage, "metrics": {"latencyMs": 1}}
    return AfterModelCallEvent(
        agent=StubAgent(),  # type: ignore[arg-type]
        stop_response=AfterModelCallEvent.ModelStopResponse(
            message=message,  # type: ignore[arg-type]
            stop_reason="end_turn",
        ),
    )


def test_usage_comes_from_the_response_not_lagging_agent_metrics() -> None:
    # Regression: Strands updates accumulated_usage only after AfterModelCallEvent, so reading
    # it there undercounted by one call (seen live: 12,706 traced vs 14,656 real tokens).
    fence = LoopFence()
    hook = fence.strands_hook()
    hook._after_model(after_model({"inputTokens": 400, "outputTokens": 100, "totalTokens": 500}))
    hook._after_model(after_model({"inputTokens": 300, "outputTokens": 50, "totalTokens": 350}))
    hook._after_model(after_model(None))  # cancelled call: no usage metadata
    report = fence.report()
    assert (report["input_tokens"], report["output_tokens"]) == (700, 150)


def test_budget_cancels_model_call() -> None:
    fence = LoopFence(FenceConfig(max_tokens=1000))
    hook = fence.strands_hook()
    agent = StubAgent()

    def before() -> BeforeModelCallEvent:
        event = BeforeModelCallEvent(agent=agent)  # type: ignore[arg-type]
        hook._before_model(event)
        return event

    assert not before().cancel
    hook._after_model(after_model({"inputTokens": 900, "outputTokens": 99, "totalTokens": 999}))
    assert not before().cancel
    hook._after_model(after_model({"inputTokens": 1, "outputTokens": 0, "totalTokens": 1}))
    stopped = before()
    assert isinstance(stopped.cancel, str) and "1,000 tokens used" in stopped.cancel
