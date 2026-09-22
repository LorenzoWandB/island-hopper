"""Evaluate one player configuration on the held-out campaign.

Weave side: a weave.Evaluation over the 10 campaign levels. Each row is played by the traced Player, so the
eval drills down to conversations (level) -> turns (attempt) -> the program and the tool result.
W&B side: one run whose config is the knobs (model, checkpoint, prompt variant, temperature, attempts) and whose
summary is the scoreboard. That run is what ARIA compares across checkpoints and configs. The run logs this
directory as a code artifact, which is what makes it a Launch job ARIA Autoresearch can re-submit with overrides.

  uv run python eval/eval_checkpoint.py --model "wandb-artifact:///<entity>/island-hopper/island-hopper-qwen14b:step10" \
      --base-url https://api.training.wandb.ai/v1/ --checkpoint 10 --name qwen14b-v3-step10
  uv run python eval/eval_checkpoint.py --model openai/gpt-oss-20b --name gpt-oss-20b-baseline
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import wandb  # noqa: E402
import weave  # noqa: E402

from islandgame import CAMPAIGN  # noqa: E402
from islandgame._env import wandb_project  # noqa: E402
from islandgame.agent import Player, PlayerConfig, PROMPT_VARIANTS, WANDB_INFERENCE_URL  # noqa: E402
from islandgame.levels import get_level  # noqa: E402
from islandgame.solver import solve  # noqa: E402

# W&B Inference list prices, USD per million tokens (input, output). Used for the cost column only.
PRICES = {
    "OpenPipe/Qwen3-14B-Instruct": (0.05, 0.22), "wandb-artifact": (0.05, 0.22),
    "openai/gpt-oss-20b": (0.05, 0.20), "openai/gpt-oss-120b": (0.15, 0.60),
    "meta-llama/Llama-3.1-8B-Instruct": (0.22, 0.22), "Qwen/Qwen3.6-35B-A3B": (0.25, 1.25),
}


def price_for(model: str) -> tuple[float, float]:
    for k, v in PRICES.items():
        if model.startswith(k):
            return v
    return (0.0, 0.0)


PROJECT = ""  # entity/project, set in main; used for the W&B Inference project header


class IslandPlayer(weave.Model):
    """Weave Model wrapper: the knobs are attributes so every eval is reproducible from its object version."""
    model_name: str
    base_url: str
    checkpoint: int
    prompt_variant: str
    temperature: float
    max_attempts: int
    tag: str

    def _player(self) -> Player:
        cfg = PlayerConfig(model=self.model_name, base_url=self.base_url, prompt_variant=self.prompt_variant,
                           temperature=self.temperature, max_attempts=self.max_attempts, tag=self.tag,
                           wandb_project=PROJECT,
                           renderer=os.environ.get("ISLAND_RENDER", "1") == "1",
                           extra={"checkpoint": self.checkpoint, "eval": True})
        return Player(cfg, trace=True)

    @weave.op()
    def predict(self, level_id: str) -> dict:
        return self._player().play_level(get_level(level_id))


@weave.op()
def solved(output: dict) -> bool:
    return bool(output["solved"])


@weave.op()
def score(output: dict) -> int:
    return int(output["final_score"])


@weave.op()
def regret(output: dict) -> int:
    """Optimal score minus achieved score; the optimum is what the solver gets."""
    return int(output["regret"])


@weave.op()
def gem_fraction(output: dict) -> float:
    return output["gems_collected"] / max(1, output["gems_total"])


@weave.op()
def attempts_used(output: dict) -> int:
    return int(output["attempts"])


@weave.op()
def fell(output: dict) -> bool:
    return any(a["status"] == "fell" for a in output["attempt_log"])


@weave.op()
def tokens(output: dict) -> int:
    return int(output["tokens"])


def _no_openai_autopatch():
    """Weave would otherwise trace the OpenAI client too, duplicating every chat span the agent tracing already records."""
    from weave.trace.autopatch import AutopatchSettings, IntegrationSettings
    return AutopatchSettings(openai=IntegrationSettings(enabled=False))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--base-url", default=WANDB_INFERENCE_URL)
    ap.add_argument("--checkpoint", type=int, default=-1, help="training step of the checkpoint, -1 for an untrained model")
    ap.add_argument("--variant", default="baseline", choices=sorted(PROMPT_VARIANTS))
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--project", default=wandb_project(), help="entity/project; defaults to WANDB_PROJECT from .env")
    ap.add_argument("--name", default=None, help="W&B run name / eval name")
    ap.add_argument("--group", default="campaign-eval", help="W&B run group")
    ap.add_argument("--no-renderer", action="store_true")
    a = ap.parse_args()
    if a.no_renderer:
        os.environ["ISLAND_RENDER"] = "0"
    global PROJECT
    PROJECT = a.project
    if "/" not in a.project:
        raise SystemExit("set --project <entity>/<project> or WANDB_PROJECT (see .env.example)")
    entity, project = a.project.split("/", 1)
    name = a.name or f"{a.model.split('/')[-1]}-{a.variant}-t{a.temperature}"

    weave.init(a.project, autopatch_settings=_no_openai_autopatch())  # manual agent spans already cover the LLM call
    run = wandb.init(entity=entity, project=project, name=name, group=a.group, job_type="campaign-eval",
                     # wandb resolves config values that start with wandb-artifact:// as artifacts, so store the name plainly
                     config={"model": a.model.replace("wandb-artifact:///", "checkpoint:"), "base_url": a.base_url, "checkpoint": a.checkpoint,
                             "prompt_variant": a.variant, "temperature": a.temperature, "max_attempts": a.attempts,
                             "levels": [lv.id for lv in CAMPAIGN]})
    run.log_code(root=str(Path(__file__).resolve().parent.parent), include_fn=lambda p: p.endswith((".py", ".toml", ".html")))

    dataset = [{"level_id": lv.id, "name": lv.name, "gems": len(lv.gems), "switches": len(lv.switches),
                "max_commands": lv.max_commands, "optimal_score": solve(lv)[1]} for lv in CAMPAIGN]
    player = IslandPlayer(name=name, model_name=a.model, base_url=a.base_url, checkpoint=a.checkpoint,
                          prompt_variant=a.variant, temperature=a.temperature, max_attempts=a.attempts, tag=name)
    evaluation = weave.Evaluation(name="campaign", dataset=dataset,
                                  scorers=[solved, score, regret, gem_fraction, attempts_used, fell, tokens],
                                  evaluation_name=name)
    t0 = time.time()
    res = await evaluation.evaluate(player)
    elapsed = time.time() - t0

    def m(scorer: str, key: str):
        return res.get(scorer, {}).get(key, {}) if isinstance(res.get(scorer, {}).get(key, {}), (int, float)) else res.get(scorer, {}).get(key, {}).get("mean")

    n = len(dataset)
    total_tokens = res["tokens"]["mean"] * n if "tokens" in res else 0
    pin, pout = price_for(a.model)
    summary = {
        "levels_solved": round(res["solved"]["true_count"]) if "solved" in res else 0,
        "solve_rate": res["solved"]["true_fraction"],
        "total_score": round(res["score"]["mean"] * n),
        "mean_score": res["score"]["mean"],
        "mean_regret": res["regret"]["mean"],
        "gem_rate": res["gem_fraction"]["mean"],
        "mean_attempts": res["attempts_used"]["mean"],
        "fall_rate": res["fell"]["true_fraction"],
        "total_tokens": total_tokens,
        "est_cost_usd": total_tokens * 0.7 * pin / 1e6 + total_tokens * 0.3 * pout / 1e6,  # rough 70/30 in/out split
        "eval_seconds": elapsed,
        "weave_project": f"https://wandb.ai/{a.project}/weave",
    }
    run.summary.update(summary)
    run.finish()
    print(f"\n{name}: solved {summary['levels_solved']}/{n}  total_score={summary['total_score']}  "
          f"mean_regret={summary['mean_regret']:.1f}  fall_rate={summary['fall_rate']:.2f}  gem_rate={summary['gem_rate']:.2f}  "
          f"tokens={int(total_tokens)}  ~${summary['est_cost_usd']:.4f}  {elapsed:.0f}s")
    print(f"W&B run: {run.url}")


if __name__ == "__main__":
    asyncio.run(main())
