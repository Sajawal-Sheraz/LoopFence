"""Data model shared by the rules, the store and the trace."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Action(StrEnum):
    ALLOW = "allow"
    BLOCK = "block"  # refuse this call; the agent may continue
    STOP = "stop"  # end the run


class Rule(StrEnum):
    """Rule ids, in evaluation order. Recorded in every block/stop trace event."""

    TIMEOUT = "timeout"
    BUDGET = "budget"
    MAX_CALLS = "max_calls"
    EXACT_REPEAT = "exact_repeat"
    POLL_MAX = "poll_max"
    POLL_UNCHANGED = "poll_unchanged"
    TOOL_CAP = "tool_cap"


RUN_LIMIT_RULES: frozenset[Rule] = frozenset({Rule.TIMEOUT, Rule.BUDGET, Rule.MAX_CALLS})


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool call about to run, with its precomputed signature."""

    tool: str
    args: Mapping[str, Any]
    signature: str
    pollable: bool = False


@dataclass(frozen=True, slots=True)
class Decision:
    """Outcome of a check. ``message`` is written for the agent when the call is refused."""

    action: Action
    rule: Rule | None
    message: str
    seq: int
    tool: str | None = None
    signature: str | None = None

    @property
    def allowed(self) -> bool:
        return self.action is Action.ALLOW


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One line of the session trace."""

    seq: int
    t: float
    kind: str  # "tool_call" | "tool_result" | "llm_call" | "usage"
    tool: str | None = None
    signature: str | None = None
    args: Any = None
    action: str | None = None
    rule: str | None = None
    message: str | None = None
    result_hash: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__ if getattr(self, k) is not None}


@dataclass(slots=True)
class SignatureState:
    calls: int = 0
    last_result: str | None = None
    unchanged_streak: int = 0  # identical results in a row, counting the first one


@dataclass(slots=True)
class SessionState:
    session_id: str
    tenant_id: str | None
    started_at: float
    total_calls: int = 0
    blocks: int = 0
    stopped_rule: Rule | None = None
    stopped_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: dict[str, int] = field(default_factory=dict)
    signatures: dict[str, SignatureState] = field(default_factory=dict)
    blocks_by_rule: dict[str, int] = field(default_factory=dict)
    events: list[TraceEvent] = field(default_factory=list)
