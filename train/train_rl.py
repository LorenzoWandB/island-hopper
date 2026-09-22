"""Serverless RL for Island Hopper with ART + W&B Serverless Training.

One trajectory = one rollout on one procedural level (optionally with a second attempt after feedback).
Reward = the engine's episode score. Groups = several rollouts on the same level, so GRPO-style
advantages compare programs for the same puzzle. Campaign levels are held out and used as `val`.

Guardrails: a hard USD budget computed from token usage at list price, a wall-clock limit, and a
step limit. Training itself is free during the preview; the budget covers inference.

  uv run python train/train_rl.py --steps 1 --groups 4 --rollouts 4      # smoke test
  uv run python train/train_rl.py --steps 40 --groups 16 --rollouts 8 --budget-usd 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import art  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402
from openai.types.chat.chat_completion import Choice  # noqa: E402

from islandgame import CAMPAIGN, generate_level, describe_level  # noqa: E402
from islandgame.engine import Episode, Level  # noqa: E402
from islandgame.agent import PROMPT_VARIANTS, parse_program, _feedback  # noqa: E402
from islandgame.solver import solve  # noqa: E402
from islandgame._env import wandb_entity_project  # noqa: E402

# W&B Inference list price for OpenPipe/Qwen3-14B-Instruct, USD per million tokens (docs, 2026-09).
PRICE_IN, PRICE_OUT = 0.05, 0.22

# Reward shaping (RL only; the game's own score is untouched). Lessons from runs 1-3:
#  run 1: invalid programs scored 0 -> model wrote over-long garbage.        (fixed in engine: -40)
#  run 2: -2/command made early falls beat late falls -> model fell early.   (progress bonus)
#  run 3: stopping short (-20) beat falling (-30) -> model stopped moving.   (this version)
# Now any non-finish costs the same, and the bonus is for getting CLOSER TO THE PORTAL along the island.
NON_FINISH_PENALTY = -30.0    # applied to fell / ended / out_of_commands / invalid alike
DISTANCE_BONUS = 6.0          # per unit of shortest-path distance to the portal closed by the program
GEM_BONUS = 20.0              # keep gems worth taking
STEP_COST = 1.0               # mild pressure toward short programs


_ENTITY, _PROJECT = wandb_entity_project()


@dataclass
class Cfg:
    entity: str = _ENTITY          # from WANDB_PROJECT=<entity>/<project> (.env); override with --entity/--project
    project: str = _PROJECT
    model_name: str = "island-hopper-qwen14b-v5"
    base_model: str = "OpenPipe/Qwen3-14B-Instruct"
    prompt_variant: str = "baseline"
    steps: int = 10
    groups_per_step: int = 8          # distinct levels per step
    rollouts_per_group: int = 12      # programs per level (needs variance within a group)
    val_rollouts: int = 3             # programs per campaign level at validation time
    attempts: int = 2                 # attempts per rollout; feedback between attempts is a non-trainable message
    learning_rate: float = 5e-6   # 1e-5 destabilized run 4 right after a curriculum promotion
    max_tokens: int = 600
    budget_usd: float = 4.0   # $5 total for the project; ~$0.75 spent on runs 1-3 and evals
    max_minutes: float = 240.0
    val_every: int = 5
    seed: int = 0
    thinking: bool = False
    start_tier: int = 1               # curriculum tier to begin at (use when resuming a model that already mastered tier 1)
    train_temperature: float = 1.2    # >1 keeps rollout diversity up; the policy collapsed to identical programs at 1.0


class Budget:
    def __init__(self, usd: float):
        self.usd, self.tokens_in, self.tokens_out = usd, 0, 0

    def add(self, usage) -> None:
        if usage:
            self.tokens_in += usage.prompt_tokens or 0
            self.tokens_out += usage.completion_tokens or 0

    @property
    def spent(self) -> float:
        return (self.tokens_in * PRICE_IN + self.tokens_out * PRICE_OUT) / 1e6

    def exhausted(self) -> bool:
        return self.spent >= self.usd


def goal_distances(level: Level) -> dict[tuple[int, int], int]:
    """Shortest-path tile distance to the portal over the island, with every bridge counted as up."""
    from collections import deque
    tiles = set(level.tiles) | {b for sw in level.switches for b in sw.bridge} | {level.goal}
    dist = {level.goal: 0}
    q = deque([level.goal])
    while q:
        x, y = q.popleft()
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            n = (x + dx, y + dy)
            if n in tiles and n not in dist:
                dist[n] = dist[(x, y)] + 1
                q.append(n)
    return dist


def closed_distance(level: Level, result) -> int:
    """How many tiles closer to the portal the program's last SAFE position is, versus the start."""
    d = goal_distances(level)
    safe = [tuple(st.pos) for st in result.steps if st.outcome != "fell"]
    last = safe[-1] if safe else tuple(level.start)
    return d.get(tuple(level.start), 0) - d.get(last, d.get(tuple(level.start), 0))


def shaped_reward(level: Level, result) -> float:
    r = GEM_BONUS * result.gems_collected - STEP_COST * result.commands_used
    if result.status == "goal":
        r += 50.0 + DISTANCE_BONUS * goal_distances(level).get(tuple(level.start), 0)
    else:
        r += NON_FINISH_PENALTY + DISTANCE_BONUS * closed_distance(level, result)
        if result.status == "invalid":
            r -= 10.0  # never let garbage beat an honest failure (run 1's lesson)
    r -= 15.0 * (result.attempt - 1)
    return float(r)


class Curriculum:
    """Adaptive: start at difficulty 1, promote when the training solve rate clears the bar, demote if it craters."""

    def __init__(self, promote_at: float = 0.3, demote_at: float = 0.05, patience: int = 2, start: int = 1):
        self.level, self.promote_at, self.demote_at, self.patience = start, promote_at, demote_at, patience
        self.streak = 0

    def sample(self, rng: random.Random) -> int:
        # mostly the current tier, some review of the tier below, a taste of the tier above
        choices = [self.level, max(1, self.level - 1), min(5, self.level + 1)]
        return rng.choices(choices, weights=[0.7, 0.15, 0.15])[0]

    def update(self, solved_rate: float) -> None:
        # promote only after `patience` consecutive qualifying steps: one lucky batch of levels is not mastery
        self.streak = self.streak + 1 if solved_rate >= self.promote_at else 0
        if self.streak >= self.patience and self.level < 5:
            self.level += 1
            self.streak = 0
        elif solved_rate <= self.demote_at and self.level > 1:
            self.level -= 1
            self.streak = 0


async def rollout(client: AsyncOpenAI, model_name: str, level: Level, cfg: Cfg, budget: Budget, temperature: float) -> art.Trajectory:
    system = PROMPT_VARIANTS[cfg.prompt_variant]
    ep = Episode(level, max_attempts=cfg.attempts)
    traj = art.Trajectory(messages_and_choices=[{"role": "system", "content": system}], reward=0.0,
                          metadata={"level_id": level.id, "difficulty": level.id.split("-")[1] if level.id.startswith("gen-") else "campaign"})
    # vLLM must return token ids so the trainer can reconstruct the exact sampled sequence.
    extra: dict = {"return_token_ids": True}
    if not cfg.thinking:
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    result = None
    for attempt in range(1, cfg.attempts + 1):
        user = describe_level(level, attempts_left=ep.attempts_left) if attempt == 1 else _feedback(result, ep.attempts_left)
        traj.messages_and_choices.append({"role": "user", "content": user})
        resp = await client.chat.completions.create(
            model=model_name, messages=traj.messages(), temperature=temperature, max_tokens=cfg.max_tokens, extra_body=extra,
        )
        budget.add(resp.usage)
        choice: Choice = resp.choices[0]
        # prompt_token_ids arrive at the response level; the trainer expects them on the Choice.
        if choice.__pydantic_extra__ is not None and "prompt_token_ids" not in choice.__pydantic_extra__:
            choice.__pydantic_extra__["prompt_token_ids"] = (resp.model_extra or {}).get("prompt_token_ids")
        traj.messages_and_choices.append(choice)
        _, program = parse_program(choice.message.content or "")
        result = ep.run(program)
        if ep.solved:
            break
    optimal_program, optimal_score = solve(level)
    final = ep.attempts[-1]
    visited = {tuple(st.pos) for st in final.steps if st.outcome != "fell"} - {tuple(level.start)}
    traj.reward = shaped_reward(level, final)
    traj.metrics = {
        "score": final.score, "reward": traj.reward, "progress": len(visited),
        "closed": closed_distance(level, final), "solved": float(ep.solved), "attempts": len(ep.attempts),
        "gems": final.gems_collected, "gem_frac": final.gems_collected / max(1, final.gems_total),
        "fell": float(final.status == "fell"), "invalid": float(final.status == "invalid"),
        "empty_program": float(final.commands_used == 0), "commands": final.commands_used,
        "regret": float(optimal_score - final.score) if ep.solved else float(optimal_score),
    }
    return traj.finish()


ROLLOUT_TIMEOUT_S = 240.0  # two attempts, each bounded by the client timeout plus retries


async def group_for(client, model_name, level, cfg, budget, n, temperature) -> art.TrajectoryGroup:
    trajs = await asyncio.gather(
        *[asyncio.wait_for(rollout(client, model_name, level, cfg, budget, temperature), timeout=ROLLOUT_TIMEOUT_S) for _ in range(n)],
        return_exceptions=True,
    )
    return art.TrajectoryGroup(trajs, metadata={"level_id": level.id})


def summarize(groups: list[art.TrajectoryGroup]) -> dict[str, float]:
    trajs = [t for g in groups for t in g.trajectories]
    if not trajs:
        return {}
    keys = ["score", "reward", "progress", "closed", "solved", "gem_frac", "fell", "invalid", "empty_program", "attempts", "regret"]
    return {k: round(sum(t.metrics.get(k, 0) for t in trajs) / len(trajs), 3) for k in keys} | {"n": len(trajs)}


async def main() -> None:
    ap = argparse.ArgumentParser()
    for f, v in asdict(Cfg()).items():
        if isinstance(v, bool):
            ap.add_argument(f"--{f.replace('_', '-')}", action="store_true", default=v)
        else:
            ap.add_argument(f"--{f.replace('_', '-')}", type=type(v), default=v)
    ap.add_argument("--groups", dest="groups_per_step", type=int)
    ap.add_argument("--rollouts", dest="rollouts_per_group", type=int)
    ap.add_argument("--no-weave", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="gather one step of rollouts and print metrics; do not train")
    ap.add_argument("--rollback-to", type=int, default=None,
                    help="delete every checkpoint above this step so training continues from it (irreversible)")
    a = ap.parse_args()
    cfg = Cfg(**{k: v for k, v in vars(a).items() if k in Cfg.__dataclass_fields__ and v is not None})
    rng = random.Random(cfg.seed)
    budget = Budget(cfg.budget_usd)
    curriculum = Curriculum(start=cfg.start_tier)
    t_start = time.time()

    if not a.no_weave:
        import weave
        weave.init(f"{cfg.entity}/{cfg.project}")

    model = art.TrainableModel(name=cfg.model_name, project=cfg.project, entity=cfg.entity, base_model=cfg.base_model)
    backend = art.ServerlessBackend()
    await model.register(backend)
    # ART's bundled client clashes with the installed openai package; a plain client works fine.
    # Hard per-request timeout: a handful of hung connections froze a whole step for 90 minutes in run 5.
    client = AsyncOpenAI(base_url=model.inference_base_url, api_key=model.inference_api_key, max_retries=2, timeout=90.0)
    if a.rollback_to is not None:
        before = await model.get_step()
        await backend._delete_checkpoint_files(model, steps_to_keep=list(range(0, a.rollback_to + 1)))
        print(f"rolled back {cfg.model_name}: latest checkpoint {before} -> {await model.get_step()}")
    start_step = await model.get_step()
    print(f"model {model.get_inference_name()}  starting at step {start_step}  budget ${cfg.budget_usd}  cfg={json.dumps(asdict(cfg))}")

    async def validate(step: int) -> dict:
        groups = await art.gather_trajectory_groups(
            [group_for(client, model.get_inference_name(), lv, cfg, budget, cfg.val_rollouts, temperature=0.3) for lv in CAMPAIGN],
            pbar_desc=f"val@{step}", max_exceptions=len(CAMPAIGN) * cfg.val_rollouts,
        )
        m = summarize(groups)
        await model.log(groups, split="val", step=step)
        print(f"  val@{step}: solved={m.get('solved')} score={m.get('score')} progress={m.get('progress')} regret={m.get('regret')} fell={m.get('fell')}")
        return m

    if start_step == 0 or a.dry_run:
        await validate(start_step)

    for i in range(cfg.steps):
        step = start_step + i
        if budget.exhausted():
            print(f"budget exhausted (${budget.spent:.2f}); stopping before step {step}"); break
        if (time.time() - t_start) / 60 > cfg.max_minutes:
            print("wall-clock limit reached; stopping"); break

        levels = [generate_level(rng.randrange(10_000_000), curriculum.sample(rng)) for _ in range(cfg.groups_per_step)]
        groups = await art.gather_trajectory_groups(
            [group_for(client, model.get_inference_name(), lv, cfg, budget, cfg.rollouts_per_group, temperature=cfg.train_temperature) for lv in levels],
            pbar_desc=f"train@{step}", max_exceptions=cfg.groups_per_step * cfg.rollouts_per_group // 4,
        )
        m = summarize(groups)
        # groups with identical rewards carry no gradient signal; drop them so training time isn't wasted
        useful = [g for g in groups if len({t.reward for t in g.trajectories}) > 1]
        tier_before = curriculum.level
        curriculum.update(m.get("solved", 0.0))
        print(f"step {step}: tier={tier_before}->{curriculum.level} rollouts={m.get('n')} solved={m.get('solved')} reward={m.get('reward')} closed={m.get('closed')} fell={m.get('fell')} "
              f"invalid={m.get('invalid')} useful_groups={len(useful)}/{len(groups)} spent=${budget.spent:.3f} "
              f"tokens={budget.tokens_in}/{budget.tokens_out} elapsed={(time.time() - t_start) / 60:.1f}m")
        await model.log(groups, split="train", step=step, metrics={"spent_usd": budget.spent, "curriculum_tier": curriculum.level, **{f"train_{k}": v for k, v in m.items()}})
        if a.dry_run:
            print("dry run: not training"); break
        if not useful:
            print("  no group had reward variance; skipping train call"); continue
        t0 = time.time()
        res = None
        for attempt in range(3):
            try:
                res = await backend.train(model, useful, learning_rate=cfg.learning_rate)
                break
            except Exception as e:  # network blips while streaming job events are common; the job itself may have finished
                await asyncio.sleep(10)
                now = await model.get_step()
                print(f"  train call failed ({type(e).__name__}: {str(e)[:120]}); server step is {now}")
                if now > step:
                    break  # the job completed server-side; carry on with the new checkpoint
                print(f"  retrying train ({attempt + 1}/3)")
        print(f"  trained -> step {await model.get_step()} in {(time.time() - t0) / 60:.1f}m  {res if res else '(recovered)'}")
        if cfg.val_every and (i + 1) % cfg.val_every == 0:
            await validate(await model.get_step())

    final_step = await model.get_step()
    if not a.dry_run and final_step != start_step:
        await validate(final_step)
    print(f"done. steps {start_step}->{final_step}  spent ${budget.spent:.3f}  ({budget.tokens_in} in / {budget.tokens_out} out tokens)  "
          f"serve as: --base-url {model.inference_base_url} --model {model.get_inference_name()}")
    await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
