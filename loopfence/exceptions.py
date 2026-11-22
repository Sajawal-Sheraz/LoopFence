"""Exceptions raised by LoopFence.

Hierarchy::

    LoopFenceError
    ├── ToolCallBlocked        soft: this one call was refused; the agent may continue
    └── RunStoppedError        hard: the whole run must end
        ├── LoopDetectedError  too many blocked calls (loop rules)
        └── BudgetExceededError  token / USD / time / total-call limit reached
"""

from __future__ import annotations


class LoopFenceError(Exception):
    """Base class. Carries the session, tool and rule that triggered it."""

    def __init__(
        self,
        reason: str,
        *,
        session_id: str,
        tool_name: str | None,
        rule: str | None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.session_id = session_id
        self.tool_name = tool_name
        self.rule = rule


class ToolCallBlocked(LoopFenceError):
    """A single tool call was refused. ``reason`` is written for the agent to read."""


class RunStoppedError(LoopFenceError):
    """The agent run must stop. Every later check in the same session also stops."""


class LoopDetectedError(RunStoppedError):
    """The run kept making calls that loop rules blocked."""


class BudgetExceededError(RunStoppedError):
    """A token, USD, wall-clock or total-call limit was reached."""
