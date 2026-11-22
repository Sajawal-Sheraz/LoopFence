"""Session summary and JSON trace export.

The exported document is versioned (``schema_version``) so recorded traces stay readable as
LoopFence evolves. With an injected clock the export is byte-identical across runs.
"""

from __future__ import annotations

import json
from typing import Any

from .config import FenceConfig
from .models import SessionState
from .rules import usd_spent

SCHEMA_VERSION = 1


def summarize(state: SessionState, config: FenceConfig) -> dict[str, Any]:
    tokens = state.input_tokens + state.output_tokens
    return {
        "session_id": state.session_id,
        "tenant_id": state.tenant_id,
        "stopped": state.stopped_rule is not None,
        "stopped_rule": state.stopped_rule.value if state.stopped_rule else None,
        "stopped_reason": state.stopped_reason,
        "allowed_calls": state.total_calls,
        "blocked_calls": state.blocks,
        "blocks_by_rule": dict(sorted(state.blocks_by_rule.items())),
        "calls_by_tool": dict(sorted(state.tool_calls.items())),
        "input_tokens": state.input_tokens,
        "output_tokens": state.output_tokens,
        "total_tokens": tokens,
        "usd": round(usd_spent(state, config), 6),
    }


def trace_document(state: SessionState, config: FenceConfig) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": state.session_id,
        "tenant_id": state.tenant_id,
        "config": config.to_dict(),
        "summary": summarize(state, config),
        "events": [event.to_dict() for event in state.events],
    }


def dumps(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
