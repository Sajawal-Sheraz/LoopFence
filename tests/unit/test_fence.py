import asyncio
import json
import threading
from typing import Any

import pytest

from loopfence import (
    Action,
    BudgetExceededError,
    FenceConfig,
    LoopDetectedError,
    LoopFence,
    Rule,
    ToolCallBlocked,
)
from tests.conftest import FakeClock

PARIS = {"origin": "NYC", "destination": "Paris", "max_price": 400}


def actions(fence: LoopFence, calls: list[tuple[str, dict[str, Any]]]) -> list[str]:
    return [fence.check(tool, args).action.value for tool, args in calls]


# ---------------------------------------------------------------- exact repeat


def test_third_identical_call_is_blocked() -> None:
    fence = LoopFence()
    assert actions(fence, [("search_flights", PARIS)] * 3) == ["allow", "allow", "block"]
    assert fence.report()["blocks_by_rule"] == {"exact_repeat": 1}


def test_block_message_tells_agent_not_to_invent_data() -> None:
    fence = LoopFence()
    for _ in range(2):
        fence.check("search_flights", PARIS)
    msg = fence.check("search_flights", PARIS).message
    assert "Reuse the earlier result" in msg
    assert "Do not invent data" in msg


def test_repeat_is_blocked_even_when_results_differ() -> None:
    # demo 3's search_flights returns random prices; that is noise, not progress.
    fence = LoopFence()
    for price in (311, 742):
        d = fence.check("search_flights", PARIS)
        fence.record_result(d, f"AirLine A: ${price}")
    assert fence.check("search_flights", PARIS).rule is Rule.EXACT_REPEAT


def test_counters_are_not_reset_between_turns() -> None:
    # One fence = one session. Turn boundaries are invisible to it by design.
    fence = LoopFence()
    fence.check("search_flights", PARIS)  # turn 1
    fence.check("search_flights", PARIS)  # turn 2
    assert fence.check("search_flights", PARIS).action is Action.BLOCK  # turn 3


# ------------------------------------------------------------ false positives


def test_distinct_destinations_are_not_blocked() -> None:
    fence = LoopFence()
    calls = [("search_flights", {**PARIS, "destination": c}) for c in ("Paris", "London", "Tokyo")]
    calls += [("check_hotel_price", {"hotel": h, "check_in": "03-15"}) for h in "ABC"]
    assert set(actions(fence, calls)) == {"allow"}


def test_polling_with_progress_is_allowed() -> None:
    # A job legitimately reports the same status several times before it changes.
    fence = LoopFence()
    statuses = ["pending"] * 2 + ["running"] * 4 + ["done"]
    for status in statuses:
        d = fence.check("get_status", {"job": 42}, pollable=True)
        assert d.allowed, d.message
        fence.record_result(d, {"status": status})


# -------------------------------------------------------------------- polling


def test_default_poll_unchanged_after_five_identical_results() -> None:
    fence = LoopFence()
    for _ in range(5):
        d = fence.check("get_status", {"job": 42}, pollable=True)
        assert d.allowed
        fence.record_result(d, "running")
    assert fence.check("get_status", {"job": 42}, pollable=True).rule is Rule.POLL_UNCHANGED


def test_poll_unchanged_with_only_timestamp_changes() -> None:
    fence = LoopFence(FenceConfig(max_unchanged_results=3))
    for second in (1, 31, 59):
        d = fence.check("get_status", {"job": 42}, pollable=True)
        assert d.allowed
        fence.record_result(d, f"pending (checked 2026-11-22T10:00:{second:02d}Z)")
    d = fence.check("get_status", {"job": 42}, pollable=True)
    assert d.rule is Rule.POLL_UNCHANGED


def test_poll_max_even_when_results_keep_changing() -> None:
    fence = LoopFence(FenceConfig(max_poll_calls=4))
    for i in range(4):
        d = fence.check("get_status", {"job": 1}, pollable=True)
        assert d.allowed
        fence.record_result(d, {"progress": i})
    assert fence.check("get_status", {"job": 1}, pollable=True).rule is Rule.POLL_MAX


def test_pollable_tools_are_exempt_from_default_tool_cap() -> None:
    fence = LoopFence(FenceConfig(max_calls_per_tool=2))
    for job in range(5):
        assert fence.check("get_status", {"job": job}, pollable=True).allowed


def test_explicit_tool_cap_applies_to_pollable_tools() -> None:
    fence = LoopFence(FenceConfig(tool_caps={"get_status": 2}))
    results = [fence.check("get_status", {"job": j}, pollable=True).action for j in range(3)]
    assert results[-1] is Action.BLOCK


# ------------------------------------------------------------------- tool cap


def test_tool_cap_blocks_ninth_call_with_distinct_args() -> None:
    fence = LoopFence()
    calls = [("check_hotel_price", {"hotel": f"H{i}"}) for i in range(9)]
    assert actions(fence, calls) == ["allow"] * 8 + ["block"]
    assert fence.report()["blocks_by_rule"] == {"tool_cap": 1}


def test_per_tool_override() -> None:
    fence = LoopFence(FenceConfig(tool_caps={"send_email": 1}))
    assert actions(fence, [("send_email", {"to": "a"}), ("send_email", {"to": "b"})]) == [
        "allow",
        "block",
    ]


# ------------------------------------------------------------ escalation/stop


def test_third_block_stops_the_run_and_stop_is_sticky() -> None:
    fence = LoopFence()
    results = actions(fence, [("search_flights", PARIS)] * 5)
    assert results == ["allow", "allow", "block", "block", "stop"]
    assert fence.stopped
    # Everything after a stop is refused, including unrelated tools and LLM calls.
    assert fence.check("other", {}).action is Action.STOP
    assert fence.check_llm_call().action is Action.STOP
    report = fence.report()
    assert report["stopped_rule"] == "exact_repeat"
    assert report["blocked_calls"] == 3


def test_max_total_calls() -> None:
    fence = LoopFence(FenceConfig(max_total_calls=3))
    calls = [("t", {"i": i}) for i in range(4)]
    assert actions(fence, calls) == ["allow", "allow", "allow", "stop"]
    assert fence.report()["stopped_rule"] == "max_calls"


def test_token_budget_stops_tools_and_llm_calls() -> None:
    fence = LoopFence(FenceConfig(max_tokens=1000))
    assert fence.check_llm_call().allowed
    fence.add_tokens(900, 50)
    assert fence.check("t", {}).allowed
    fence.add_tokens(40, 10)
    d = fence.check_llm_call()
    assert d.action is Action.STOP and d.rule is Rule.BUDGET
    assert fence.check("t", {"x": 1}).rule is Rule.BUDGET


def test_usd_budget() -> None:
    cfg = FenceConfig(max_usd=0.01, usd_per_1m_input_tokens=1.0, usd_per_1m_output_tokens=4.0)
    fence = LoopFence(cfg)
    fence.add_tokens(6000, 1000)  # $0.006 + $0.004 = $0.01
    assert fence.check_llm_call().rule is Rule.BUDGET
    assert fence.report()["usd"] == pytest.approx(0.01)


def test_timeout_with_injected_clock(clock: FakeClock) -> None:
    fence = LoopFence(FenceConfig(max_seconds=60), clock=clock)
    assert fence.check("t", {}).allowed
    clock.advance(60)
    d = fence.check("t", {"x": 1})
    assert d.action is Action.STOP and d.rule is Rule.TIMEOUT


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        FenceConfig(max_repeats=0)
    with pytest.raises(ValueError):
        FenceConfig(max_tokens=0)
    with pytest.raises(TypeError):
        FenceConfig().tool_caps["x"] = 1  # type: ignore[index]


# ------------------------------------------------------------------ decorator


def test_protect_sync_binds_positional_and_keyword_the_same() -> None:
    fence = LoopFence()
    runs: list[str] = []

    @fence.protect
    def search_flights(origin: str, destination: str, max_price: float = 500.0) -> str:
        runs.append(destination)
        return "ok"

    search_flights("NYC", "Paris", 400)
    search_flights(origin="NYC", destination="Paris", max_price=400.0)
    with pytest.raises(ToolCallBlocked) as exc:
        search_flights("NYC", destination="Paris", max_price=400)
    assert runs == ["Paris", "Paris"]
    assert exc.value.rule == "exact_repeat"
    assert exc.value.tool_name == "search_flights"
    assert exc.value.session_id == fence.session_id


def test_protect_applies_defaults() -> None:
    fence = LoopFence()

    @fence.protect
    def f(x: int, y: int = 1) -> int:
        return x + y

    f(1)
    f(1, 1)
    with pytest.raises(ToolCallBlocked):
        f(x=1)


def test_protect_hard_stop_raises_loop_detected() -> None:
    fence = LoopFence(FenceConfig(max_blocks=1))

    @fence.protect
    def f() -> None:
        return None

    f()
    f()
    with pytest.raises(LoopDetectedError):
        f()


def test_protect_budget_raises_budget_exceeded() -> None:
    fence = LoopFence(FenceConfig(max_tokens=10))

    @fence.protect(name="lookup")
    def f(q: str) -> str:
        return q

    fence.add_tokens(10)
    with pytest.raises(BudgetExceededError) as exc:
        f("x")
    assert exc.value.tool_name == "lookup"


def test_protect_async_and_pollable() -> None:
    fence = LoopFence(FenceConfig(max_unchanged_results=3))

    @fence.protect(pollable=True)
    async def get_status(job: int) -> str:
        return "pending"

    async def run() -> None:
        for _ in range(3):
            assert await get_status(7) == "pending"
        with pytest.raises(ToolCallBlocked) as exc:
            await get_status(7)
        assert exc.value.rule == "poll_unchanged"

    asyncio.run(run())


def test_protect_records_exceptions_as_results() -> None:
    fence = LoopFence(FenceConfig(max_unchanged_results=3))

    @fence.protect(pollable=True)
    def flaky() -> None:
        raise RuntimeError("503 Service Unavailable")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            flaky()
    with pytest.raises(ToolCallBlocked) as exc:
        flaky()
    assert exc.value.rule == "poll_unchanged"


# ---------------------------------------------------------------- concurrency


def test_concurrent_checks_keep_counters_exact() -> None:
    fence = LoopFence(
        FenceConfig(
            max_repeats=10_000, max_calls_per_tool=10_000, max_total_calls=None, max_blocks=10_000
        )
    )
    allowed: list[bool] = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(200):
            ok = fence.check("t", {"same": True}).allowed
            with lock:
                allowed.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    report = fence.report()
    assert report["allowed_calls"] == sum(allowed) == 1600


def test_concurrent_checks_never_exceed_repeat_limit() -> None:
    fence = LoopFence(FenceConfig(max_repeats=5, max_blocks=10_000))
    results: list[bool] = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(50):
            ok = fence.check("t", {"same": True}).allowed
            with lock:
                results.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(results) == 5


# ---------------------------------------------------------------------- trace


def _run_session(clock: FakeClock) -> LoopFence:
    fence = LoopFence(FenceConfig(max_tokens=50_000), session_id="s1", clock=clock)
    for i in range(4):
        clock.advance(1.5)
        d = fence.check("search_flights", PARIS)
        if d.allowed:
            fence.record_result(d, f"price {i}")
        fence.add_tokens(1000 * (i + 1), 100)
    return fence


def test_trace_counts_match_reality(clock: FakeClock) -> None:
    fence = _run_session(clock)
    doc = json.loads(fence.export_trace())
    calls = [e for e in doc["events"] if e["kind"] == "tool_call"]
    assert doc["schema_version"] == 1
    assert sum(e["action"] == "allow" for e in calls) == doc["summary"]["allowed_calls"] == 2
    refused = [e for e in calls if e["action"] != "allow"]
    assert len(refused) == doc["summary"]["blocked_calls"] == 2
    assert all(e["rule"] and e["message"] for e in refused)
    assert [e["seq"] for e in doc["events"]] == list(range(len(doc["events"])))


def test_trace_is_byte_identical_across_runs(tmp_path: Any) -> None:
    a = _run_session(FakeClock()).export_trace()
    b = _run_session(FakeClock()).export_trace(tmp_path / "trace.json")
    assert a == b == (tmp_path / "trace.json").read_text()
