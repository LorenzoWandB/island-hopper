"""Load a `.env` file from the project root into os.environ, without overriding what is already set.

Keeps account-specific settings (W&B entity, checkpoint names) out of the code. Copy `.env.example`
to `.env` and fill it in; `.env` is gitignored.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key.startswith("export "):
            key = key[len("export "):].strip()
        os.environ.setdefault(key, value)


def wandb_project() -> str:
    """`entity/project` for Weave, W&B runs, and Serverless Training. Empty if not configured."""
    return os.environ.get("WANDB_PROJECT", "")


def wandb_entity_project() -> tuple[str, str]:
    ep = wandb_project()
    if "/" not in ep:
        raise SystemExit("set WANDB_PROJECT=<entity>/<project> in .env or the environment (see .env.example)")
    entity, project = ep.split("/", 1)
    return entity, project


load_dotenv()
