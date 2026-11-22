"""LoopFence configuration: every limit in one frozen object."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

DEFAULT_VOLATILE_KEYS: frozenset[str] = frozenset(
    {
        "request_id",
        "requestid",
        "trace_id",
        "traceid",
        "span_id",
        "nonce",
        "timestamp",
        "ts",
        "created_at",
        "updated_at",
        "toolUseId",
        "tool_use_id",
    }
)


@dataclass(frozen=True, slots=True, kw_only=True)
class FenceConfig:
    """Limits for one agent run. ``None`` disables a limit.

    Counting convention: ``max_*`` is the number of things *allowed*; the next one is refused.
    """

    # Loop rules (soft blocks)
    max_repeats: int = 2
    """Identical calls (same tool + normalized args) allowed for a normal tool."""

    max_calls_per_tool: int = 8
    """Calls allowed per tool, any args. Pollable tools are exempt unless in ``tool_caps``."""

    tool_caps: Mapping[str, int] = field(default_factory=dict)
    """Per-tool override of ``max_calls_per_tool``; also applies to pollable tools."""

    max_poll_calls: int = 20
    """Identical calls allowed for a pollable tool."""

    max_unchanged_results: int = 5
    """Pollable tool: after this many identical results in a row, the next call is blocked."""

    max_blocks: int = 3
    """The Nth block in a session becomes a hard stop."""

    # Run limits (hard stops)
    max_total_calls: int | None = 50
    max_tokens: int | None = None
    max_usd: float | None = None
    usd_per_1m_input_tokens: float = 0.0
    usd_per_1m_output_tokens: float = 0.0
    max_seconds: float | None = None

    # Normalization
    volatile_keys: frozenset[str] = DEFAULT_VOLATILE_KEYS
    """Dict keys dropped before hashing args and results."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_caps", MappingProxyType(dict(self.tool_caps)))
        for name in (
            "max_repeats",
            "max_calls_per_tool",
            "max_poll_calls",
            "max_unchanged_results",
            "max_blocks",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        for tool, cap in self.tool_caps.items():
            if cap < 0:
                raise ValueError(f"tool_caps[{tool!r}] must be >= 0")
        for name in ("max_total_calls", "max_tokens", "max_usd", "max_seconds"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0 or None")

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly view, used in trace exports."""
        return {
            "max_repeats": self.max_repeats,
            "max_calls_per_tool": self.max_calls_per_tool,
            "tool_caps": dict(sorted(self.tool_caps.items())),
            "max_poll_calls": self.max_poll_calls,
            "max_unchanged_results": self.max_unchanged_results,
            "max_blocks": self.max_blocks,
            "max_total_calls": self.max_total_calls,
            "max_tokens": self.max_tokens,
            "max_usd": self.max_usd,
            "usd_per_1m_input_tokens": self.usd_per_1m_input_tokens,
            "usd_per_1m_output_tokens": self.usd_per_1m_output_tokens,
            "max_seconds": self.max_seconds,
            "volatile_keys": sorted(self.volatile_keys),
        }
