"""Strands Agents adapter. Requires ``pip install loopfence[strands]``.

Mapping:

* ``BeforeToolCallEvent``  -> ``fence.check``; block via ``cancel_tool`` (the model sees the
  message as the tool's error result); stop via ``agent.cancel(after_current_tools=True)``.
* ``AfterToolCallEvent``   -> ``fence.record_result`` (status + content; ``toolUseId`` dropped).
* ``BeforeModelCallEvent`` -> ``fence.check_llm_call``; stop via ``cancel`` (ends the turn with
  our message as the final answer).
* ``AfterModelCallEvent``  -> ``fence.add_tokens`` from the response's usage metadata.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from strands.hooks import (
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
)

from ..models import Action, Decision

if TYPE_CHECKING:
    from ..fence import LoopFence


class LoopFenceHook(HookProvider):
    """Attach with ``Agent(..., hooks=[fence.strands_hook()])``."""

    def __init__(self, fence: LoopFence, *, pollable_tools: Iterable[str] = ()) -> None:
        self.fence = fence
        self.pollable_tools = frozenset(pollable_tools)
        self._lock = threading.Lock()
        self._pending: dict[str, Decision] = {}

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self._before_tool)
        registry.add_callback(AfterToolCallEvent, self._after_tool)
        registry.add_callback(BeforeModelCallEvent, self._before_model)
        registry.add_callback(AfterModelCallEvent, self._after_model)

    def _before_tool(self, event: BeforeToolCallEvent) -> None:
        tool_use = event.tool_use
        name = str(tool_use["name"])
        raw_args = tool_use.get("input")
        args = raw_args if isinstance(raw_args, dict) else {"input": raw_args}
        decision = self.fence.check(name, args, pollable=name in self.pollable_tools)
        if decision.allowed:
            with self._lock:
                self._pending[str(tool_use["toolUseId"])] = decision
            return
        event.cancel_tool = decision.message
        if decision.action is Action.STOP:
            event.agent.cancel(message=decision.message, after_current_tools=True)

    def _after_tool(self, event: AfterToolCallEvent) -> None:
        with self._lock:
            decision = self._pending.pop(str(event.tool_use["toolUseId"]), None)
        if decision is None:
            return
        result = event.result
        self.fence.record_result(
            decision, {"status": result.get("status"), "content": result.get("content")}
        )

    def _before_model(self, event: BeforeModelCallEvent) -> None:
        decision = self.fence.check_llm_call()
        if not decision.allowed:
            event.cancel = decision.message

    def _after_model(self, event: AfterModelCallEvent) -> None:
        # Read this call's usage from the response itself. Strands only adds it to
        # ``event_loop_metrics.accumulated_usage`` *after* this hook runs.
        if event.stop_response is None:
            return
        message: Any = event.stop_response.message
        usage = (message.get("metadata") or {}).get("usage") or {}
        self.fence.add_tokens(int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0)))
