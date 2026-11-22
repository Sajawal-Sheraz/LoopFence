"""Deterministic normalization and hashing of tool arguments and results.

Pure functions: no I/O, no global state.

Arguments are normalized *structurally* only (volatile keys dropped, keys sorted, integral
floats folded to ints) so that distinct requests never collapse into one signature.

Results additionally get *text masking*: timestamps, UUIDs and long hex IDs become fixed
placeholders, so a stuck tool whose output only differs by an embedded clock value is still
recognized as unchanged.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any

_DATETIME = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?"
)
_UUID = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_LONG_HEX = re.compile(r"\b[0-9a-fA-F]{16,}\b")


def mask_text(text: str) -> str:
    """Replace timestamps, UUIDs and long hex strings with fixed placeholders.

    Bare dates such as ``2026-03-15`` are kept: they are usually content (a check-in date),
    not noise.
    """
    text = _DATETIME.sub("<TS>", text)
    text = _UUID.sub("<UUID>", text)
    return _LONG_HEX.sub("<HEX>", text)


def normalize(value: Any, volatile_keys: frozenset[str], *, mask: bool = False) -> Any:
    """Return a JSON-serializable, order-independent form of ``value``."""
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        return value
    if isinstance(value, str):
        return mask_text(value) if mask else value
    if isinstance(value, Mapping):
        return {
            str(k): normalize(v, volatile_keys, mask=mask)
            for k, v in value.items()
            if str(k) not in volatile_keys
        }
    if isinstance(value, list | tuple):
        return [normalize(v, volatile_keys, mask=mask) for v in value]
    if isinstance(value, set | frozenset):
        items = [normalize(v, volatile_keys, mask=mask) for v in value]
        return sorted(items, key=_dumps)
    if isinstance(value, bytes):
        return {"__bytes_sha256__": hashlib.sha256(value).hexdigest()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return normalize(dataclasses.asdict(value), volatile_keys, mask=mask)
    model_dump = getattr(value, "model_dump", None)  # pydantic v2, without importing it
    if callable(model_dump):
        return normalize(model_dump(), volatile_keys, mask=mask)
    # Last resort. repr() may embed a memory address; such objects hash per-instance.
    return f"<{type(value).__qualname__}:{value!r}>"


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_json(value: Any, volatile_keys: frozenset[str], *, mask: bool = False) -> str:
    """Compact, key-sorted JSON of the normalized value."""
    return _dumps(normalize(value, volatile_keys, mask=mask))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def call_signature(tool: str, args: Mapping[str, Any], volatile_keys: frozenset[str]) -> str:
    """SHA-256 over ``(tool, normalized args)``. Same logical call -> same signature."""
    return _sha256(tool + "\n" + canonical_json(args, volatile_keys))


def result_hash(result: Any, volatile_keys: frozenset[str]) -> str:
    """SHA-256 over a normalized, masked result.

    A string that parses as a JSON object or array is hashed as structured data, so volatile
    keys inside serialized JSON are dropped too.
    """
    if isinstance(result, str):
        stripped = result.strip()
        if stripped[:1] in ("{", "["):
            with contextlib.suppress(ValueError):
                result = json.loads(stripped)
    return _sha256(canonical_json(result, volatile_keys, mask=True))
