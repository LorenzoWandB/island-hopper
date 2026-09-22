"""Hand-built campaign (the held-out test set) and a procedural generator (the training set)."""
from __future__ import annotations

import random
from .engine import Level, Switch, Pos, DIR_VEC, LEFT, RIGHT


def _path(*pts: Pos) -> set[Pos]:
    """Tiles along axis-aligned segments between consecutive waypoints, inclusive."""
    tiles: set[Pos] = set()
    for a, b in zip(pts, pts[1:]):
        (x0, y0), (x1, y1) = a, b
        assert x0 == x1 or y0 == y1, "waypoints must be axis-aligned"
        sx = (x1 > x0) - (x1 < x0)
        sy = (y1 > y0) - (y1 < y0)
        x, y = x0, y0
        tiles.add((x, y))
        while (x, y) != (x1, y1):
            x, y = x + sx, y + sy
            tiles.add((x, y))
    if len(pts) == 1:
        tiles.add(pts[0])
    return tiles


def _ring(x0: int, y0: int, x1: int, y1: int) -> set[Pos]:
    return {(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)
            if x in (x0, x1) or y in (y0, y1)}


CAMPAIGN: list[Level] = [
    Level(
        id="1", name="First Steps",
        tiles=_path((0, 0), (4, 0)), start=(0, 0), facing="E", goal=(4, 0),
        gems={(2, 0)}, max_commands=8,
        hint="Walk east, grab the gem on the way, keep walking.",
    ),
    Level(
        id="2", name="The Ring",
        tiles=_ring(0, 0, 3, 3), start=(0, 0), facing="E", goal=(3, 3),
        gems={(3, 0), (0, 3)}, max_commands=16,
        hint="Both gems are worth the detour.",
    ),
    Level(
        id="3", name="Zigzag",
        tiles=_path((0, 0), (0, 2), (2, 2), (2, 4), (4, 4)), start=(0, 0), facing="S", goal=(4, 4),
        gems={(0, 2), (2, 4)}, max_commands=16,
    ),
    Level(
        id="4", name="Dead End",
        tiles=_path((0, 2), (3, 2)) | _path((2, 2), (2, 0)) | _path((3, 2), (3, 4)),
        start=(0, 2), facing="E", goal=(3, 4),
        gems={(2, 0)}, max_commands=16,
        hint="The gem is up a dead-end branch. You will have to turn around.",
    ),
    Level(
        id="5", name="The Gap",
        tiles=_path((0, 0), (2, 0)) | _path((4, 0), (6, 0)), start=(0, 0), facing="E", goal=(6, 0),
        gems={(5, 0)}, switches=[Switch(pos=(2, 0), bridge=[(3, 0)])], max_commands=12,
        hint="Stand on the switch and toggle it before you cross.",
    ),
    Level(
        id="6", name="Detour Switch",
        tiles=_path((0, 1), (2, 1)) | _path((2, 1), (2, 3)) | _path((4, 1), (5, 1)),
        start=(0, 1), facing="E", goal=(5, 1),
        gems={(2, 3)}, switches=[Switch(pos=(2, 3), bridge=[(3, 1)])], max_commands=18,
        hint="The switch is on a side branch, with a gem on it.",
    ),
    Level(
        id="7", name="Spiral",
        tiles=_path((0, 0), (4, 0), (4, 4), (0, 4), (0, 2), (2, 2)), start=(0, 0), facing="E", goal=(2, 2),
        gems={(4, 0), (0, 4)}, max_commands=22,
    ),
    Level(
        id="8", name="Double Bridge",
        tiles=_path((0, 0), (1, 0)) | _path((3, 0), (3, 2)) | _path((5, 2), (6, 2)),
        start=(0, 0), facing="E", goal=(6, 2),
        gems={(3, 0)}, switches=[Switch(pos=(1, 0), bridge=[(2, 0), (4, 2)])], max_commands=16,
        hint="One switch, two bridges.",
    ),
    Level(
        id="9", name="Long Haul",
        tiles=_path((0, 0), (3, 0), (3, 3), (6, 3), (6, 0)) | _path((3, 3), (3, 5)),
        start=(0, 0), facing="E", goal=(6, 0),
        gems={(3, 0), (3, 5), (6, 3)}, max_commands=22,
        hint="Three gems, tight budget. Decide which detours pay for themselves.",
    ),
    Level(
        id="10", name="Grand Finale",
        tiles=_ring(0, 0, 2, 2) | _path((2, 1), (3, 1)) | _path((5, 1), (7, 1), (7, 4)) | _path((6, 4), (7, 4)),
        start=(0, 0), facing="E", goal=(6, 4),
        gems={(0, 2), (7, 1), (5, 1)}, switches=[Switch(pos=(2, 2), bridge=[(4, 1)])], max_commands=25,
        hint="Everything at once: a loop, a switch, a gap, and gems on both sides of it.",
    ),
]

_BY_ID = {lv.id: lv for lv in CAMPAIGN}


def get_level(level_id: str) -> Level:
    if level_id in _BY_ID:
        return _BY_ID[level_id]
    if level_id.startswith("gen-"):
        # gen-<difficulty>-<seed>
        _, diff, seed = level_id.split("-", 2)
        return generate_level(int(seed), difficulty=int(diff))
    raise KeyError(f"unknown level {level_id!r}. Campaign ids: {list(_BY_ID)}; procedural: gen-<difficulty 1-5>-<seed>")


def generate_level(seed: int, difficulty: int = 2) -> Level:
    """Procedural level, solvable by construction: a self-avoiding random walk with gems on it,
    optionally one gap bridged by a switch placed earlier on the path. Verified with the solver."""
    from .solver import solve

    difficulty = max(1, min(5, difficulty))
    rng = random.Random(seed * 7919 + difficulty)
    for _ in range(200):
        length = {1: 5, 2: 7, 3: 10, 4: 13, 5: 16}[difficulty] + rng.randint(0, 2)
        n_gems = {1: 1, 2: 2, 3: 2, 4: 3, 5: 3}[difficulty]
        use_switch = difficulty >= 3 and rng.random() < 0.7
        turn_bias = 0.25 + 0.1 * difficulty

        pos: Pos = (0, 0)
        facing = rng.choice(list(DIR_VEC))
        start_facing = facing
        path: list[Pos] = [pos]
        ok = True
        for _ in range(length):
            # Prefer going straight; sometimes turn. Never revisit or touch an existing tile diagonally-adjacent-in-line.
            options = [facing] + ([LEFT[facing], RIGHT[facing]] if rng.random() < turn_bias else [])
            rng.shuffle(options)
            moved = False
            for f in options:
                dx, dy = DIR_VEC[f]
                nxt = (pos[0] + dx, pos[1] + dy)
                # keep the walk self-avoiding and not adjacent to older path tiles (keeps the island a clean corridor)
                if nxt in path:
                    continue
                neighbors = [(nxt[0] + ddx, nxt[1] + ddy) for ddx, ddy in DIR_VEC.values()]
                if any(n in path[:-1] for n in neighbors):
                    continue
                pos, facing = nxt, f
                path.append(pos)
                moved = True
                break
            if not moved:
                ok = False
                break
        if not ok or len(path) < 4:
            continue

        tiles = set(path)
        goal = path[-1]
        switches: list[Switch] = []
        if use_switch and len(path) >= 7:
            gap_i = rng.randint(3, len(path) - 3)
            gap = path[gap_i]
            sw_i = rng.randint(1, gap_i - 1)
            tiles.discard(gap)
            switches.append(Switch(pos=path[sw_i], bridge=[gap]))
        candidates = [p for p in path[1:-1] if p in tiles and all(p != s.pos for s in switches)]
        rng.shuffle(candidates)
        gems = set(candidates[:n_gems])
        max_commands = len(path) + 2 * n_gems + 8 + 2 * len(switches)
        level = Level(
            id=f"gen-{difficulty}-{seed}", name=f"Procedural D{difficulty} #{seed}",
            tiles=tiles, start=path[0], facing=start_facing, goal=goal,
            gems=gems, switches=switches, max_commands=max_commands,
        )
        program, _ = solve(level)
        if program is not None:
            return level
    raise RuntimeError(f"could not generate a solvable level for seed={seed} difficulty={difficulty}")
