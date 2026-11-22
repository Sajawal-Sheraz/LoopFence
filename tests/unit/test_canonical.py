from dataclasses import dataclass

from loopfence.canonical import call_signature, mask_text, result_hash
from loopfence.config import DEFAULT_VOLATILE_KEYS as V


def test_key_order_does_not_matter() -> None:
    a = call_signature("t", {"a": 1, "b": {"x": 1, "y": 2}}, V)
    b = call_signature("t", {"b": {"y": 2, "x": 1}, "a": 1}, V)
    assert a == b


def test_integral_float_equals_int() -> None:
    assert call_signature("t", {"max_price": 400}, V) == call_signature(
        "t", {"max_price": 400.0}, V
    )
    assert call_signature("t", {"p": 400.5}, V) != call_signature("t", {"p": 400}, V)


def test_volatile_keys_are_dropped_recursively() -> None:
    a = call_signature("t", {"q": "x", "request_id": "1", "meta": {"timestamp": 1, "k": 2}}, V)
    b = call_signature("t", {"q": "x", "request_id": "2", "meta": {"timestamp": 9, "k": 2}}, V)
    assert a == b


def test_tool_name_and_values_distinguish_calls() -> None:
    assert call_signature("a", {"x": 1}, V) != call_signature("b", {"x": 1}, V)
    assert call_signature("a", {"city": "Paris"}, V) != call_signature("a", {"city": "London"}, V)


def test_args_are_not_text_masked() -> None:
    # Distinct IDs in arguments are distinct requests and must not collapse.
    a = call_signature("get", {"id": "3f2b8c1e-0000-4000-8000-000000000001"}, V)
    b = call_signature("get", {"id": "3f2b8c1e-0000-4000-8000-000000000002"}, V)
    assert a != b


def test_mask_text_placeholders() -> None:
    text = (
        "pending at 2026-11-22T10:00:01Z job 3f2b8c1e-0000-4000-8000-000000000001 "
        "commit deadbeefdeadbeef01 check-in 2026-03-15"
    )
    assert mask_text(text) == "pending at <TS> job <UUID> commit <HEX> check-in 2026-03-15"


def test_result_hash_ignores_timestamps_in_text() -> None:
    a = result_hash("status: pending (checked 2026-11-22T10:00:01Z)", V)
    b = result_hash("status: pending (checked 2026-11-22T10:00:31Z)", V)
    c = result_hash("status: running (checked 2026-11-22T10:01:01Z)", V)
    assert a == b != c


def test_result_hash_ignores_volatile_keys_in_json_and_json_strings() -> None:
    a = result_hash({"status": "pending", "updated_at": "x"}, V)
    b = result_hash('{"updated_at": "y", "status": "pending"}', V)
    assert a == b


def test_dataclass_and_set_are_deterministic() -> None:
    @dataclass
    class Q:
        city: str
        tags: frozenset[str]

    a = call_signature("t", {"q": Q("Paris", frozenset({"b", "a"}))}, V)
    b = call_signature("t", {"q": Q("Paris", frozenset({"a", "b"}))}, V)
    assert a == b
