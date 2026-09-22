"""The RL reward must strictly prefer: finish > get closer > stop short ~ fall early; never reward timidity."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from train.train_rl import shaped_reward, closed_distance
from islandgame import get_level
from islandgame.engine import run_program


def test_reward_ordering_on_first_steps():
    lv = get_level("1")  # straight line of 5, gem at (2,0), portal at (4,0)
    finish = shaped_reward(lv, run_program(lv, ["move_forward", "move_forward", "collect_gem", "move_forward", "move_forward"]))
    far_then_fall = shaped_reward(lv, run_program(lv, ["move_forward", "move_forward", "move_forward", "turn_left", "move_forward"]))
    timid = shaped_reward(lv, run_program(lv, ["move_forward"]))
    do_nothing = shaped_reward(lv, run_program(lv, ["turn_left"]))
    fall_now = shaped_reward(lv, run_program(lv, ["turn_left", "move_forward"]))
    garbage = shaped_reward(lv, run_program(lv, ["fly"]))
    assert finish > far_then_fall > timid > do_nothing >= fall_now
    assert garbage <= fall_now


def test_closed_distance_counts_last_safe_tile():
    lv = get_level("1")
    assert closed_distance(lv, run_program(lv, ["move_forward"] * 3 + ["turn_left", "move_forward"])) == 3
