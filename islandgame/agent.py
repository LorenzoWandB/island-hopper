"""The player: an LLM that writes a whole program for a level, runs it, reads the result, retries.

Tracing follows Weave's agent model:
  conversation = one level (all attempts)      turn = one attempt
  chat span    = the model writing the program  execute_tool span = run_program on the engine

Model access is any OpenAI-compatible endpoint; defaults to W&B Inference. `--mock` swaps the
model for a scripted policy so the whole pipeline (tracing, renderer, eval) runs with no API key.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass, field, asdict
from typing import Any

from .engine import Level, Episode, RunResult, COMMANDS, describe_level
from .levels import CAMPAIGN, get_level
from .renderer_client import emit
from .solver import solve

WANDB_INFERENCE_URL = "https://api.inference.wandb.ai/v1"

PROMPT_VARIANTS: dict[str, str] = {
    "baseline": (
        "You are playing Island Hopper, a puzzle game. Read the level, then write a program: a list of commands "
        "from move_forward, turn_left, turn_right, collect_gem, toggle_switch. "
        "Respond with JSON only: {\"thought\": \"<one sentence plan>\", \"program\": [\"move_forward\", ...]}."
    ),
    "careful": (
        "You are playing Island Hopper, a puzzle game on floating tiles. Stepping onto water is fatal, so before each "
        "move_forward check that the tile in front of you exists (or is a bridge you have already raised). "
        "Track your position and facing after every command. Gems are worth 20, commands cost 2 each: a gem is "
        "worth up to a 10-command detour. Respond with JSON only: {\"thought\": \"<one sentence plan>\", "
        "\"program\": [\"move_forward\", ...]}."
    ),
    "planner": (
        "You are an expert at Island Hopper. Solve it in two phases inside your head: (1) list the waypoints you "
        "will visit in order (start, gems worth taking, switches you need, portal) with coordinates; (2) translate the "
        "waypoint path into commands, simulating position and facing after each one. x grows east, y grows south. "
        "Facing N means y decreases with move_forward. turn_left from N gives W; turn_right from N gives E. "
        "Respond with JSON only: {\"thought\": \"<waypoints>\", \"program\": [\"move_forward\", ...]}."
    ),
}


@dataclass
class PlayerConfig:
    model: str = os.environ.get("ISLAND_MODEL") or "openai/gpt-oss-20b"   # your own model via .env, else W&B Inference
    temperature: float = 0.3
    prompt_variant: str = "baseline"
    max_attempts: int = 3
    max_tokens: int = 4000
    reasoning_effort: str = "low"   # applied to openai/gpt-oss-* models only
    thinking: bool = False          # Qwen3 checkpoints: enable the thinking chat template
    base_url: str = os.environ.get("ISLAND_BASE_URL") or WANDB_INFERENCE_URL
    wandb_project: str = os.environ.get("WANDB_PROJECT", "")
    mock: str = ""            # "" | "solver" | "sloppy" | "random"
    renderer: bool = True
    tag: str = ""             # groups conversations, e.g. "ckpt-0"
    extra: dict[str, Any] = field(default_factory=dict)


# ---------- program parsing ----------

_CMD_RE = re.compile(r"\b(" + "|".join(COMMANDS) + r")\b")


def parse_program(text: str) -> tuple[str, list[str]]:
    """Accept JSON, fenced JSON, or free text; fall back to command names in order of appearance."""
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            prog = [str(c).strip().strip("()") for c in obj.get("program", [])]
            return str(obj.get("thought", ""))[:300], prog
        except json.JSONDecodeError:
            pass
    return text.splitlines()[0][:300] if text else "", _CMD_RE.findall(text)


# ---------- mock policies (no model needed) ----------

def mock_program(kind: str, level: Level, attempt: int, rng: random.Random) -> tuple[str, list[str]]:
    optimal, _ = solve(level)
    optimal = optimal or ["move_forward"]
    if kind == "solver":
        return "optimal program from the solver", optimal
    if kind == "sloppy":
        # gets it right on the last attempt; earlier attempts drop a command
        if attempt >= 2 or rng.random() < 0.3:
            return "second look, fixed the missing turn", optimal
        bad = optimal[:]
        k = rng.randrange(len(bad))
        del bad[k]
        return f"first guess (accidentally skipped step {k + 1})", bad
    n = rng.randint(3, min(level.max_commands, 10))
    return "random walk", [rng.choice(["move_forward", "move_forward", "turn_left", "turn_right", "collect_gem"]) for _ in range(n)]


# ---------- tracing shim ----------

class Tracer:
    """Thin wrapper so the play loop reads the same with Weave on or off."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        if enabled:
            import weave  # noqa
            self.weave = weave

    def conversation(self, **kw):
        return self.weave.start_conversation(**kw) if self.enabled else nullcontext(_Null())

    def turn(self, conv, **kw):
        return conv.start_turn(**kw) if self.enabled else nullcontext(_Null())


class _Null:
    def __getattr__(self, _):
        return lambda *a, **k: self

    def __setattr__(self, k, v):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ---------- the player ----------

class Player:
    def __init__(self, cfg: PlayerConfig, trace: bool = True):
        self.cfg = cfg
        self.tracer = Tracer(trace)
        self.rng = random.Random(0)
        self.client = None
        if not cfg.mock:
            from openai import OpenAI
            headers = {}
            if cfg.base_url == WANDB_INFERENCE_URL and cfg.wandb_project:
                headers["OpenAI-Project"] = cfg.wandb_project
            # W&B endpoints use the W&B key; anything else (OpenAI, Ollama, vLLM, ...) uses ISLAND_API_KEY / OPENAI_API_KEY,
            # falling back to a dummy value for local servers that do not check it.
            api_key = (os.environ.get("WANDB_API_KEY") if "wandb.ai" in cfg.base_url
                       else os.environ.get("ISLAND_API_KEY") or os.environ.get("OPENAI_API_KEY", "x"))
            if not api_key and "wandb.ai" in cfg.base_url:  # fall back to the stored wandb login
                try:
                    import netrc
                    api_key = netrc.netrc(os.path.expanduser("~/.netrc")).authenticators("api.wandb.ai")[2]
                except Exception:
                    api_key = None
            self.client = OpenAI(base_url=cfg.base_url, api_key=api_key or "missing", default_headers=headers)

    # one model call = one chat span
    def _generate(self, turn, messages: list[dict]) -> tuple[str, list[str], dict]:
        cfg = self.cfg
        t0 = time.time()
        with turn.start_llm(model=cfg.model, provider_name="wandb-inference" if cfg.base_url == WANDB_INFERENCE_URL else "openai-compatible") as llm:
            kwargs: dict[str, Any] = {}
            if cfg.model.startswith("openai/gpt-oss") and cfg.reasoning_effort:
                kwargs["extra_body"] = {"reasoning_effort": cfg.reasoning_effort}
            if cfg.model.startswith("wandb-artifact://") or "Qwen3-14B" in cfg.model:
                kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": cfg.thinking}}
            try:
                resp = self.client.chat.completions.create(
                    model=cfg.model, messages=messages, temperature=cfg.temperature, max_tokens=cfg.max_tokens, **kwargs,
                )
            except Exception as e:  # recorded on the span, then re-raised so the turn shows the error
                llm.record_error(e)
                raise
            msg = resp.choices[0].message
            reasoning = (msg.model_extra or {}).get("reasoning") or (msg.model_extra or {}).get("reasoning_content") or ""
            text = msg.content or ""
            truncated = resp.choices[0].finish_reason == "length" and not text
            usage = getattr(resp, "usage", None)
            usage_d = {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens} if usage else {}
            llm.set_attributes({"latency_s": round(time.time() - t0, 3), "temperature": cfg.temperature})
            if self.tracer.enabled:
                from weave.conversation import Usage, Message  # type: ignore
                llm.record(
                    input_messages=[Message(role=m["role"], content=m["content"]) for m in messages],
                    output_messages=[Message(role="assistant", content=text)],
                    usage=Usage(input_tokens=usage_d.get("prompt_tokens", 0), output_tokens=usage_d.get("completion_tokens", 0)) if usage_d else None,
                    reasoning=reasoning or None,
                    response_model=getattr(resp, "model", None),
                    finish_reasons=[resp.choices[0].finish_reason] if resp.choices[0].finish_reason else None,
                )
        if truncated:  # spent the whole budget thinking: an empty program, not a scraped one
            return "(ran out of tokens while reasoning; no program produced)", [], {"raw": reasoning[-500:], "truncated": True, **usage_d, "latency_s": round(time.time() - t0, 3)}
        thought, program = parse_program(text)
        return thought, program, {"raw": text, "reasoning": reasoning[:2000], **usage_d, "latency_s": round(time.time() - t0, 3)}

    def play_level(self, level: Level) -> dict[str, Any]:
        cfg = self.cfg
        ep = Episode(level, max_attempts=cfg.max_attempts)
        system = PROMPT_VARIANTS[cfg.prompt_variant]
        conv_id = f"{cfg.tag + '-' if cfg.tag else ''}{level.id}-{int(time.time())}"
        history: list[dict] = [{"role": "system", "content": system}]
        attempts_out: list[dict] = []
        tokens = 0

        with self.tracer.conversation(
            agent_name="island-hopper", model=cfg.model, conversation_id=conv_id,
            conversation_name=f"Level {level.id}: {level.name}",
            attributes={"level_id": level.id, "prompt_variant": cfg.prompt_variant, "tag": cfg.tag, **cfg.extra},
        ) as conv:
            if cfg.renderer:
                emit({"type": "level", "level": level.to_dict(), "attempt": 1, "source": "agent"})
            while not ep.solved and ep.attempts_left > 0:
                attempt = len(ep.attempts) + 1
                user = describe_level(level, attempts_left=ep.attempts_left) if attempt == 1 else _feedback(ep.attempts[-1], ep.attempts_left)
                history.append({"role": "user", "content": user})
                with self.tracer.turn(conv, user_message=user, model=cfg.model, agent_name="island-hopper", system_instructions=[system]) as turn:
                    if cfg.mock:
                        thought, program = mock_program(cfg.mock, level, attempt, self.rng)
                        meta: dict[str, Any] = {}
                    else:
                        thought, program, meta = self._generate(turn, history)
                        tokens += meta.get("prompt_tokens", 0) + meta.get("completion_tokens", 0)
                    history.append({"role": "assistant", "content": json.dumps({"thought": thought, "program": program})})

                    if cfg.renderer:
                        emit({"type": "level", "level": level.to_dict(), "attempt": attempt, "source": "agent"})
                        emit({"type": "program", "commands": program, "thought": thought, "source": "agent"})

                    tool = turn.start_tool(name="run_program", arguments=json.dumps({"level_id": level.id, "commands": program}))
                    result = ep.run(program)
                    tool.result = json.dumps({"status": result.status, "score": result.score, "message": result.message,
                                              "gems": f"{result.gems_collected}/{result.gems_total}"})
                    tool.end()
                    if cfg.renderer:
                        emit({"type": "run", "result": result.to_dict(), "source": "agent"})

                    if self.tracer.enabled:
                        from weave.conversation import Message  # type: ignore
                        turn.record(output_messages=[Message(role="assistant", content=json.dumps({"thought": thought, "program": program}))])
                    turn.set_attributes({"attempt": attempt, "status": result.status, "score": result.score,
                                         "gems_collected": result.gems_collected, "commands_used": result.commands_used})
                    attempts_out.append({"attempt": attempt, "thought": thought, "program": program,
                                         "status": result.status, "score": result.score, "message": result.message, **meta})

        best = ep.best_score()
        optimal_program, optimal_score = solve(level)
        return {
            "level_id": level.id, "level_name": level.name, "solved": ep.solved, "attempts": len(ep.attempts),
            "best_score": best, "final_score": ep.attempts[-1].score if ep.attempts else 0,
            "optimal_score": optimal_score, "regret": (optimal_score - best) if ep.solved else optimal_score,
            "gems_collected": max(a.gems_collected for a in ep.attempts) if ep.attempts else 0, "gems_total": len(level.gems),
            "tokens": tokens, "conversation_id": conv_id, "attempt_log": attempts_out,
        }

    def play_levels(self, levels: list[Level]) -> dict[str, Any]:
        per_level = [self.play_level(lv) for lv in levels]
        n = len(per_level)
        return {
            "config": asdict(self.cfg),
            "levels_solved": sum(r["solved"] for r in per_level), "levels_total": n,
            "total_score": sum(r["final_score"] for r in per_level),
            "mean_attempts": round(sum(r["attempts"] for r in per_level) / n, 2),
            "mean_regret": round(sum(r["regret"] for r in per_level) / n, 2),
            "gem_rate": round(sum(r["gems_collected"] for r in per_level) / max(1, sum(r["gems_total"] for r in per_level)), 3),
            "total_tokens": sum(r["tokens"] for r in per_level),
            "per_level": per_level,
        }


def _feedback(result: RunResult, attempts_left: int) -> str:
    log = "\n".join(
        f"  {s.index + 1}. {s.command} -> {s.outcome}" + (f" ({s.note})" if s.note else "") + f", now at {list(s.pos)} facing {s.facing}"
        for s in result.steps
    )
    if not result.steps and result.commands_used == 0:
        return ("Your last response contained no program. Respond with JSON only: "
                "{\"thought\": \"...\", \"program\": [\"move_forward\", ...]}. "
                f"You have {attempts_left} attempt(s) left.")
    return (
        f"Run result: {result.status.upper()}. {result.message}\n"
        f"Score {result.score}. Gems {result.gems_collected}/{result.gems_total}.\n"
        f"Step log:\n{log}\n\n"
        f"You have {attempts_left} attempt(s) left. Fix the program. Respond with JSON only."
    )


def parse_levels(spec: str) -> list[Level]:
    out: list[Level] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part and not part.startswith("gen"):
            a, b = part.split("-")
            out += [get_level(str(i)) for i in range(int(a), int(b) + 1)]
        else:
            out.append(get_level(part))
    return out


def _no_openai_autopatch():
    """Weave would otherwise trace the OpenAI client too, duplicating every chat span the agent tracing already records."""
    from weave.trace.autopatch import AutopatchSettings, IntegrationSettings
    return AutopatchSettings(openai=IntegrationSettings(enabled=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Play Island Hopper with an LLM (or a mock policy), traced in Weave.")
    ap.add_argument("--levels", default="1-10", help="e.g. 1-10 or 2,5,gen-3-7")
    ap.add_argument("--model", default=PlayerConfig.model)
    ap.add_argument("--temperature", type=float, default=PlayerConfig.temperature)
    ap.add_argument("--variant", default="baseline", choices=sorted(PROMPT_VARIANTS))
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=PlayerConfig.max_tokens)
    ap.add_argument("--reasoning-effort", default=PlayerConfig.reasoning_effort, choices=["", "low", "medium", "high"])
    ap.add_argument("--thinking", action="store_true", help="Qwen3 checkpoints: enable thinking")
    ap.add_argument("--base-url", default=PlayerConfig.base_url, help="any OpenAI-compatible endpoint (ISLAND_BASE_URL)")
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", ""), help="entity/project for Weave and W&B Inference")
    ap.add_argument("--mock", default="", choices=["", "solver", "sloppy", "random"])
    ap.add_argument("--no-weave", action="store_true", help="skip Weave tracing (automatic when WANDB_PROJECT is not set)")
    ap.add_argument("--no-renderer", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--json", default="", help="write the summary here")
    a = ap.parse_args()

    trace = not a.no_weave and bool(a.project)
    if not a.no_weave and not a.project:
        print("no WANDB_PROJECT set: playing without Weave tracing (set it in .env to trace games)")
    if trace:
        import weave
        weave.init(a.project, autopatch_settings=_no_openai_autopatch())  # manual agent spans already cover the LLM call

    cfg = PlayerConfig(model=a.model, temperature=a.temperature, prompt_variant=a.variant, max_attempts=a.attempts,
                       base_url=a.base_url, wandb_project=a.project, mock=a.mock, renderer=not a.no_renderer, tag=a.tag,
                       max_tokens=a.max_tokens, reasoning_effort=a.reasoning_effort, thinking=a.thinking)
    player = Player(cfg, trace=trace)
    summary = player.play_levels(parse_levels(a.levels))

    for r in summary["per_level"]:
        mark = "✓" if r["solved"] else "✗"
        print(f"{mark} L{r['level_id']:<3} {r['level_name']:<15} attempts={r['attempts']} score={r['final_score']:>4} optimal={r['optimal_score']:>3}  {r['attempt_log'][-1]['thought'][:60]}")
    print(f"\nsolved {summary['levels_solved']}/{summary['levels_total']}  total_score={summary['total_score']}  "
          f"mean_attempts={summary['mean_attempts']}  mean_regret={summary['mean_regret']}  gem_rate={summary['gem_rate']}  tokens={summary['total_tokens']}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
