"""Fallback: supervised warm-start from the solver, then RL on top.

The solver produces the optimal program for any procedural level for free. Fine-tuning on a few hundred of
those teaches the base model the response format, full-length programs, and how turning changes facing.
RL (train_rl.py --model-name <this model>) then continues from the SFT checkpoint.

  uv run python train/sft_warmstart.py --n 600 --dry-run          # build + inspect the dataset, no cost
  uv run python train/sft_warmstart.py --n 600 --run              # upload + train (Serverless SFT; free in preview)
  uv run python train/train_rl.py --model-name island-hopper-qwen14b-sft --steps 40 ...   # then RL from it
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import art  # noqa: E402

from islandgame import generate_level, describe_level  # noqa: E402
from islandgame.engine import Level, DIR_VEC, LEFT, RIGHT, run_program  # noqa: E402
from islandgame.agent import PROMPT_VARIANTS  # noqa: E402
from islandgame.solver import solve  # noqa: E402
from islandgame._env import wandb_entity_project  # noqa: E402

SEED_BASE = 5_000_000  # far from the RL generator's seed range so SFT and RL levels never overlap


def narrate(level: Level, program: list[str]) -> str:
    """A short, truthful plan for the program, so the model learns to think in waypoints before writing code."""
    pos, facing = level.start, level.facing
    parts: list[str] = []
    run = 0
    for cmd in program + ["<end>"]:
        if cmd == "move_forward":
            run += 1
            dx, dy = DIR_VEC[facing]
            pos = (pos[0] + dx, pos[1] + dy)
            continue
        if run:
            parts.append(f"walk {run} {facing} to {list(pos)}")
            run = 0
        if cmd == "turn_left":
            facing = LEFT[facing]; parts.append(f"turn left to face {facing}")
        elif cmd == "turn_right":
            facing = RIGHT[facing]; parts.append(f"turn right to face {facing}")
        elif cmd == "collect_gem":
            parts.append("collect the gem")
        elif cmd == "toggle_switch":
            parts.append("toggle the switch to raise the bridge")
    parts[-1] = parts[-1] + " (portal)" if parts else "portal"
    return f"Start at {list(level.start)} facing {level.facing}: " + "; ".join(parts) + "."


def build_dataset(n: int, difficulties: tuple[int, ...] = (1, 1, 2, 2, 3, 4, 5), seed: int = 0) -> list[art.Trajectory]:
    rng = random.Random(seed)
    system = PROMPT_VARIANTS["baseline"]
    out: list[art.Trajectory] = []
    seen: set[str] = set()
    i = 0
    while len(out) < n:
        d = difficulties[i % len(difficulties)]
        lv = generate_level(SEED_BASE + i, d)
        i += 1
        program, score = solve(lv)
        if not program:
            continue
        res = run_program(lv, program)
        assert res.status == "goal", (lv.id, res.message)
        key = describe_level(lv)
        if key in seen:
            continue
        seen.add(key)
        answer = json.dumps({"thought": narrate(lv, program), "program": program})
        out.append(art.Trajectory(
            messages_and_choices=[
                {"role": "system", "content": system},
                {"role": "user", "content": describe_level(lv, attempts_left=3)},
                {"role": "assistant", "content": answer},
            ],
            reward=float(score),
            metadata={"level_id": lv.id, "difficulty": str(d)},
            metrics={"program_len": len(program), "gems": len(lv.gems), "switches": len(lv.switches)},
        ))
    return out


_ENTITY, _PROJECT = wandb_entity_project()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--entity", default=_ENTITY, help="defaults to WANDB_PROJECT=<entity>/<project> from .env")
    ap.add_argument("--project", default=_PROJECT)
    ap.add_argument("--model-name", default="island-hopper-qwen14b-sft")
    ap.add_argument("--base-model", default="OpenPipe/Qwen3-14B-Instruct")
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--out", default="train/data/sft_solver.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()

    trajs = build_dataset(a.n)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for t in trajs:
            f.write(json.dumps({"messages": t.messages(), "metadata": t.metadata}) + "\n")
    lens = [t.metrics["program_len"] for t in trajs]
    turns = sum(1 for t in trajs if any(c in ("turn_left", "turn_right") for c in json.loads(t.messages()[-1]["content"])["program"]))
    print(f"{len(trajs)} examples -> {a.out}")
    print(f"difficulty mix: {dict(sorted(Counter(t.metadata['difficulty'] for t in trajs).items()))}")
    print(f"program length: min {min(lens)} mean {sum(lens)/len(lens):.1f} max {max(lens)}; with turns: {turns}/{len(trajs)}; "
          f"with switches: {sum(1 for t in trajs if t.metrics['switches'])}")
    print("\nexample:\n" + trajs[3].messages()[-1]["content"][:400])
    if a.dry_run or not a.run:
        print("\n(dry run; pass --run to upload and train)")
        return

    model = art.TrainableModel(name=a.model_name, project=a.project, entity=a.entity, base_model=a.base_model)
    backend = art.ServerlessBackend()
    await model.register(backend)
    print(f"registered {model.get_inference_name()} at step {await model.get_step()}; starting SFT on {len(trajs)} examples")
    await model.train_sft(trajs, config=art.TrainSFTConfig(learning_rate=a.learning_rate), verbose=True)
    print(f"SFT done -> step {await model.get_step()}. Next: eval it, then RL from it:\n"
          f"  uv run python eval/eval_checkpoint.py --model {model.get_inference_name()} --base-url {model.inference_base_url} --checkpoint 0 --name {a.model_name}-sft\n"
          f"  uv run python train/train_rl.py --model-name {a.model_name} --steps 40 --groups 16 --rollouts 12 --val-every 10 --budget-usd 2")
    await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
