"""The rule pipeline. Pure: functions read or mutate only the ``SessionState`` they are given.

Callers (``LoopFence``) run these inside ``SessionStore.update`` so each check-and-record is
atomic per session.

Order: run limits first (timeout, budget, max_calls -> STOP), then loop rules
(exact_repeat / poll_max / poll_unchanged / tool_cap -> BLOCK). The ``max_blocks``-th block
escalates to STOP. Once stopped, a session stays stopped.
"""

from __future__ import annotations

from .config import FenceConfig
from .models import Action, Rule, SessionState, SignatureState, ToolCall

_NO_INVENTING = "Do not invent data that no tool returned."
_STOPPED = "LoopFence stopped this run."


def usd_spent(state: SessionState, config: FenceConfig) -> float:
    return (
        state.input_tokens * config.usd_per_1m_input_tokens
        + state.output_tokens * config.usd_per_1m_output_tokens
    ) / 1_000_000


def _run_limit(state: SessionState, config: FenceConfig, now: float) -> tuple[Rule, str] | None:
    """Hard limits that apply to both tool calls and LLM calls."""
    elapsed = now - state.started_at
    if config.max_seconds is not None and elapsed >= config.max_seconds:
        return Rule.TIMEOUT, f"run time {elapsed:.1f}s reached the {config.max_seconds:g}s limit"
    tokens = state.input_tokens + state.output_tokens
    if config.max_tokens is not None and tokens >= config.max_tokens:
        return Rule.BUDGET, f"{tokens:,} tokens used, limit {config.max_tokens:,}"
    if config.max_usd is not None:
        usd = usd_spent(state, config)
        if usd >= config.max_usd:
            return Rule.BUDGET, f"${usd:.4f} spent, limit ${config.max_usd:.4f}"
    return None


def _loop_rule(state: SessionState, call: ToolCall, config: FenceConfig) -> tuple[Rule, str] | None:
    sig = state.signatures.get(call.signature)
    if call.pollable:
        if sig is not None and sig.calls >= config.max_poll_calls:
            return Rule.POLL_MAX, (
                f"{call.tool} has been polled with these arguments {sig.calls} times "
                f"(limit {config.max_poll_calls}). Stop polling and report the current status."
            )
        if sig is not None and sig.unchanged_streak >= config.max_unchanged_results:
            return Rule.POLL_UNCHANGED, (
                f"{call.tool} returned the same result {sig.unchanged_streak} times in a row; "
                "it is not making progress. Stop polling and report the current status."
            )
    elif sig is not None and sig.calls >= config.max_repeats:
        return Rule.EXACT_REPEAT, (
            f"{call.tool} was already called with these exact arguments {sig.calls} times. "
            f"Reuse the earlier result instead of calling it again. {_NO_INVENTING}"
        )

    cap = config.tool_caps.get(call.tool)
    if cap is None and not call.pollable:
        cap = config.max_calls_per_tool
    used = state.tool_calls.get(call.tool, 0)
    if cap is not None and used >= cap:
        return Rule.TOOL_CAP, (
            f"{call.tool} has been called {used} times (limit {cap}). Work with the results you "
            f"already have; if they are not enough, tell the user. {_NO_INVENTING}"
        )
    return None


def decide(
    state: SessionState, call: ToolCall, config: FenceConfig, now: float
) -> tuple[Action, Rule | None, str]:
    """Decide whether ``call`` may run. Does not mutate ``state``."""
    if state.stopped_rule is not None:
        return (
            Action.STOP,
            state.stopped_rule,
            _STOPPED if state.stopped_reason is None else state.stopped_reason,
        )
    limit = _run_limit(state, config, now)
    if limit is not None:
        rule, reason = limit
        return Action.STOP, rule, f"LoopFence stopped this run: {reason}."
    if config.max_total_calls is not None and state.total_calls >= config.max_total_calls:
        return (
            Action.STOP,
            Rule.MAX_CALLS,
            (
                f"LoopFence stopped this run: {state.total_calls} tool calls made "
                f"(limit {config.max_total_calls})."
            ),
        )

    loop = _loop_rule(state, call, config)
    if loop is None:
        return Action.ALLOW, None, ""
    rule, reason = loop
    if state.blocks + 1 >= config.max_blocks:
        return (
            Action.STOP,
            rule,
            (f"LoopFence stopped this run after {state.blocks + 1} blocked calls. Last: {reason}"),
        )
    return Action.BLOCK, rule, f"LoopFence blocked this call: {reason}"


def decide_llm_call(
    state: SessionState, config: FenceConfig, now: float
) -> tuple[Action, Rule | None, str]:
    """Decide whether another LLM call may run (time and budget only)."""
    if state.stopped_rule is not None:
        return (
            Action.STOP,
            state.stopped_rule,
            _STOPPED if state.stopped_reason is None else state.stopped_reason,
        )
    limit = _run_limit(state, config, now)
    if limit is None:
        return Action.ALLOW, None, ""
    rule, reason = limit
    return Action.STOP, rule, f"LoopFence stopped this run: {reason}."


def apply(
    state: SessionState, call: ToolCall | None, action: Action, rule: Rule | None, message: str
) -> None:
    """Update counters for a decision returned by ``decide`` / ``decide_llm_call``."""
    if action is Action.ALLOW:
        if call is None:
            return
        state.total_calls += 1
        state.tool_calls[call.tool] = state.tool_calls.get(call.tool, 0) + 1
        state.signatures.setdefault(call.signature, SignatureState()).calls += 1
        return
    assert rule is not None
    already_stopped = state.stopped_rule is not None
    if not already_stopped and call is not None:
        state.blocks += 1
        state.blocks_by_rule[rule.value] = state.blocks_by_rule.get(rule.value, 0) + 1
    if action is Action.STOP and not already_stopped:
        state.stopped_rule = rule
        state.stopped_reason = message


def record_result(state: SessionState, signature: str, result_digest: str) -> None:
    """Track consecutive identical results for a signature."""
    sig = state.signatures.setdefault(signature, SignatureState())
    if sig.last_result == result_digest:
        sig.unchanged_streak += 1
    else:
        sig.last_result = result_digest
        sig.unchanged_streak = 1
