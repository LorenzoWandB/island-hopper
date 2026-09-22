from islandgame import CAMPAIGN, get_level, generate_level, describe_level
from islandgame.engine import run_program, Episode
from islandgame.solver import solve


def test_campaign_all_solvable_and_optimal_program_runs():
    for lv in CAMPAIGN:
        program, score = solve(lv)
        assert program is not None, f"level {lv.id} unsolvable"
        assert len(program) <= lv.max_commands
        res = run_program(lv, program)
        assert res.status == "goal", (lv.id, res.message)
        assert res.score == score, (lv.id, res.score, score)


def test_fall_and_wasted():
    lv = get_level("1")
    res = run_program(lv, ["turn_left", "move_forward"])
    assert res.status == "fell"
    res = run_program(lv, ["collect_gem"])
    assert res.steps[0].outcome == "wasted"


def test_bridge_requires_switch():
    lv = get_level("5")
    res = run_program(lv, ["move_forward", "move_forward", "move_forward"])
    assert res.status == "fell"
    res = run_program(lv, ["move_forward", "move_forward", "toggle_switch", "move_forward", "move_forward"])
    assert res.status != "fell" and res.final_pos == (4, 0)


def test_episode_attempt_penalty():
    ep = Episode(get_level("1"))
    ep.run(["turn_left", "move_forward"])
    r2 = ep.run(["move_forward"] * 4)
    assert r2.attempt == 2 and r2.status == "goal"
    assert r2.score == 50 - 8 - 15
    assert ep.solved


def test_procedural_levels_generate_and_describe():
    for d in range(1, 6):
        for seed in range(5):
            lv = generate_level(seed, d)
            assert solve(lv)[0] is not None
            assert "LEVEL" in describe_level(lv)


def test_no_safe_harbor_for_rl():
    """Invalid and do-nothing programs must score worse than an honest attempt that falls early."""
    lv = get_level("2")
    fell = run_program(lv, ["turn_left", "move_forward"]).score
    invalid = run_program(lv, ["move_forward"] * 99).score
    unknown = run_program(lv, ["fly"]).score
    empty = run_program(lv, []).score
    assert invalid < fell and unknown < fell
    assert empty < 0
