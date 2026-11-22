"""Replay the three runs recorded from demo 3 on qwen3:14b through a fresh fence.

The recorded sequences are what the agent did *without* a fence; replaying them shows where
the fence would have intervened. (With a live fence the agent reacts to the block message, so
the later calls would differ.)
"""

from typing import Any

from loopfence import FenceConfig, LoopFence


def replay(trial: list[dict[str, Any]], config: FenceConfig | None = None) -> list[dict[str, Any]]:
    """Return one record per recorded call: index (1-based), turn, tool, action, rule."""
    fence = LoopFence(config)
    out: list[dict[str, Any]] = []
    seen_tokens = 0
    n = 0
    for turn_no, turn in enumerate(trial, 1):
        for tool, args in turn["calls"]:
            n += 1
            d = fence.check(tool, args)
            out.append(
                {"n": n, "turn": turn_no, "tool": tool, "action": d.action.value, "rule": d.rule}
            )
            if d.allowed:
                fence.record_result(d, "recorded result")
        fence.add_tokens(turn["tokens_cumulative"] - seen_tokens)
        seen_tokens = turn["tokens_cumulative"]
    return out


def first_refused(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((r for r in records if r["action"] != "allow"), None)


def test_trial_1_exact_repeat_blocked_at_third_identical_search(demo3: dict[str, Any]) -> None:
    records = replay(demo3["trials"]["trial_1"])
    first = first_refused(records)
    assert first == {
        "n": 5,
        "turn": 3,
        "tool": "search_flights",
        "action": "block",
        "rule": "exact_repeat",
    }
    # Unfenced the agent made 6 identical searches; the fence lets 2 through, then stops the run.
    searches = [r for r in records if r["tool"] == "search_flights"]
    assert [r["action"] for r in searches] == ["allow", "allow", "block", "block", "stop", "stop"]


def test_trial_2_tool_cap_blocks_ninth_hotel_check(demo3: dict[str, Any]) -> None:
    records = replay(demo3["trials"]["trial_2"])
    hotels = [r for r in records if r["tool"] == "check_hotel_price"]
    assert len(hotels) == 10
    assert [r["action"] for r in hotels[:8]] == ["allow"] * 8
    assert hotels[8]["rule"] == "tool_cap"
    assert hotels[8]["action"] == "block"


def test_trial_3_is_not_caught_by_loop_rules(demo3: dict[str, Any]) -> None:
    # Documented v1 limitation: 7 distinct invented hotel names stay under the tool cap.
    records = replay(demo3["trials"]["trial_3"])
    assert first_refused(records) is None


def test_trial_3_is_stopped_by_token_budget(demo3: dict[str, Any]) -> None:
    records = replay(demo3["trials"]["trial_3"], FenceConfig(max_tokens=7_000))
    first = first_refused(records)
    assert first is not None
    assert (first["turn"], first["action"], first["rule"]) == (3, "stop", "budget")
