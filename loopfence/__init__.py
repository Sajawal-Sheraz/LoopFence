"""LoopFence: stop an AI-agent run from looping or overspending, and record why it stopped."""

from .config import FenceConfig
from .exceptions import (
    BudgetExceededError,
    LoopDetectedError,
    LoopFenceError,
    RunStoppedError,
    ToolCallBlocked,
)
from .fence import LoopFence
from .models import Action, Decision, Rule

__all__ = [
    "Action",
    "BudgetExceededError",
    "Decision",
    "FenceConfig",
    "LoopDetectedError",
    "LoopFence",
    "LoopFenceError",
    "Rule",
    "RunStoppedError",
    "ToolCallBlocked",
]
__version__ = "0.1.0"
