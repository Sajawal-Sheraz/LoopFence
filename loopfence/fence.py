"""``LoopFence``: the public facade. One instance guards one agent run (one session)."""

from __future__ import annotations

import functools
import inspect
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar, overload

from . import rules
from .canonical import call_signature, normalize, result_hash
from .config import FenceConfig
from .exceptions import BudgetExceededError, LoopDetectedError, RunStoppedError, ToolCallBlocked
from .models import RUN_LIMIT_RULES, Action, Decision, SessionState, ToolCall, TraceEvent
from .store import InMemoryStore, SessionStore
from .trace import dumps, summarize, trace_document

_SKIP_PARAMS = frozenset({"self", "cls"})
_F = TypeVar("_F", bound=Callable[..., Any])


class LoopFence:
    """Guards one agent run against loops and overspending.

    Integration points:

    * ``@fence.protect`` / ``@fence.protect(pollable=True)`` on plain tool functions, or
    * ``fence.check()`` + ``fence.record_result()`` from a framework hook, and
    * ``fence.check_llm_call()`` + ``fence.add_tokens()`` around LLM calls.
    """

    def __init__(
        self,
        config: FenceConfig | None = None,
        *,
        session_id: str | None = None,
        tenant_id: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        store: SessionStore | None = None,
    ) -> None:
        self.config = config or FenceConfig()
        self.session_id = session_id or uuid.uuid4().hex
        self.tenant_id = tenant_id
        self._clock = clock
        self._store: SessionStore = store or InMemoryStore()
        self._started_at = clock()
        self._store.create(
            SessionState(
                session_id=self.session_id, tenant_id=tenant_id, started_at=self._started_at
            )
        )

    # ------------------------------------------------------------------ checks

    def check(
        self, tool: str, args: Mapping[str, Any] | None = None, *, pollable: bool = False
    ) -> Decision:
        """Decide whether a tool call may run, and record the decision. Never raises."""
        args = dict(args or {})
        call = ToolCall(
            tool=tool,
            args=args,
            signature=call_signature(tool, args, self.config.volatile_keys),
            pollable=pollable,
        )
        trace_args = normalize(args, self.config.volatile_keys)
        now = self._clock()

        def step(state: SessionState) -> Decision:
            action, rule, message = rules.decide(state, call, self.config, now)
            rules.apply(state, call, action, rule, message)
            seq = len(state.events)
            state.events.append(
                TraceEvent(
                    seq=seq,
                    t=self._elapsed(now),
                    kind="tool_call",
                    tool=tool,
                    signature=call.signature,
                    args=trace_args,
                    action=action.value,
                    rule=rule.value if rule else None,
                    message=message or None,
                )
            )
            return Decision(action, rule, message, seq, tool, call.signature)

        return self._store.update(self.session_id, step)

    def record_result(self, decision: Decision, result: Any) -> None:
        """Record what an allowed call returned. Used to detect unchanged polling results."""
        if not decision.allowed or decision.signature is None:
            return
        digest = result_hash(result, self.config.volatile_keys)
        signature = decision.signature
        now = self._clock()

        def step(state: SessionState) -> None:
            rules.record_result(state, signature, digest)
            state.events.append(
                TraceEvent(
                    seq=len(state.events),
                    t=self._elapsed(now),
                    kind="tool_result",
                    tool=decision.tool,
                    signature=signature,
                    result_hash=digest,
                )
            )

        self._store.update(self.session_id, step)

    def check_llm_call(self) -> Decision:
        """Decide whether another LLM call may run (time and budget). Never raises."""
        now = self._clock()

        def step(state: SessionState) -> Decision:
            action, rule, message = rules.decide_llm_call(state, self.config, now)
            rules.apply(state, None, action, rule, message)
            seq = len(state.events)
            state.events.append(
                TraceEvent(
                    seq=seq,
                    t=self._elapsed(now),
                    kind="llm_call",
                    action=action.value,
                    rule=rule.value if rule else None,
                    message=message or None,
                )
            )
            return Decision(action, rule, message, seq)

        return self._store.update(self.session_id, step)

    def add_tokens(self, input_tokens: int, output_tokens: int = 0) -> None:
        """Add LLM token usage. Limits are enforced at the next check."""
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be >= 0")
        if input_tokens == 0 and output_tokens == 0:
            return
        now = self._clock()

        def step(state: SessionState) -> None:
            state.input_tokens += input_tokens
            state.output_tokens += output_tokens
            state.events.append(
                TraceEvent(
                    seq=len(state.events),
                    t=self._elapsed(now),
                    kind="usage",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            )

        self._store.update(self.session_id, step)

    # ----------------------------------------------------------------- results

    @property
    def stopped(self) -> bool:
        return self._store.snapshot(self.session_id).stopped_rule is not None

    def report(self) -> dict[str, Any]:
        """Counters and stop reason for this session."""
        return summarize(self._store.snapshot(self.session_id), self.config)

    def export_trace(self, path: str | Path | None = None) -> str:
        """Return the JSON trace; also write it to ``path`` if given."""
        text = dumps(trace_document(self._store.snapshot(self.session_id), self.config))
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def raise_for(self, decision: Decision) -> None:
        """Raise the matching exception for a refused decision; no-op if allowed."""
        if decision.allowed:
            return
        rule = decision.rule.value if decision.rule else None
        kwargs: dict[str, Any] = {
            "session_id": self.session_id,
            "tool_name": decision.tool,
            "rule": rule,
        }
        if decision.action is Action.BLOCK:
            raise ToolCallBlocked(decision.message, **kwargs)
        if decision.rule in RUN_LIMIT_RULES:
            raise BudgetExceededError(decision.message, **kwargs)
        if decision.rule is not None:
            raise LoopDetectedError(decision.message, **kwargs)
        raise RunStoppedError(decision.message, **kwargs)

    # -------------------------------------------------------------- decorator

    @overload
    def protect(self, fn: _F, /) -> _F: ...

    @overload
    def protect(
        self, fn: None = None, /, *, pollable: bool = False, name: str | None = None
    ) -> Callable[[_F], _F]: ...

    def protect(
        self,
        fn: Callable[..., Any] | None = None,
        /,
        *,
        pollable: bool = False,
        name: str | None = None,
    ) -> Any:
        """Decorate a tool function (``def`` or ``async def``).

        A refused call raises ``ToolCallBlocked`` (soft) or a ``RunStoppedError`` subclass
        (hard). Agent frameworks usually turn tool exceptions into a message for the model,
        which is exactly what a soft block wants.
        """

        def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or func.__name__
            signature = inspect.signature(func)

            def bind(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                return {k: v for k, v in bound.arguments.items() if k not in _SKIP_PARAMS}

            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    decision = self.check(tool_name, bind(args, kwargs), pollable=pollable)
                    self.raise_for(decision)
                    try:
                        result = await func(*args, **kwargs)
                    except Exception as exc:
                        self.record_result(decision, _error_result(exc))
                        raise
                    self.record_result(decision, result)
                    return result

                return async_wrapper

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                decision = self.check(tool_name, bind(args, kwargs), pollable=pollable)
                self.raise_for(decision)
                try:
                    result = func(*args, **kwargs)
                except Exception as exc:
                    self.record_result(decision, _error_result(exc))
                    raise
                self.record_result(decision, result)
                return result

            return wrapper

        if fn is not None:
            return decorate(fn)
        return decorate

    # ----------------------------------------------------------------- helpers

    def _elapsed(self, now: float) -> float:
        return round(now - self._started_at, 6)


def _error_result(exc: Exception) -> dict[str, str]:
    return {"error": type(exc).__name__, "message": str(exc)}


__all__ = ["LoopFence"]
