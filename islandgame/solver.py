"""Optimal-program solver. Used to verify levels are solvable and to compute a reference score.

State = (pos, facing, gems_remaining, switches_on). Dijkstra on command count, then take the
best-scoring goal state. Small levels only (this is a puzzle game, not chess).
"""
from __future__ import annotations

import heapq
from .engine import (
    Level, COMMANDS, DIR_VEC, LEFT, RIGHT,
    SCORE_GOAL, SCORE_GEM, SCORE_PER_COMMAND,
)


def solve(level: Level) -> tuple[list[str] | None, int]:
    """Return (best_program, best_score) or (None, 0) if the portal is unreachable within max_commands."""
    gem_list = sorted(level.gems)
    gem_index = {g: i for i, g in enumerate(gem_list)}
    start = (level.start, level.facing, (1 << len(gem_list)) - 1, 0)
    dist: dict[tuple, int] = {start: 0}
    prev: dict[tuple, tuple[tuple, str]] = {}
    pq = [(0, 0, start)]
    tie = 1
    best: tuple[int, tuple] | None = None

    def walkable(p, sw_mask):
        if p in level.tiles:
            return True
        for k, sw in enumerate(level.switches):
            if (sw_mask >> k) & 1 and p in sw.bridge:
                return True
        return False

    while pq:
        d, _, st = heapq.heappop(pq)
        if d != dist.get(st):
            continue
        if d >= level.max_commands:
            continue
        pos, facing, gmask, smask = st
        for cmd in COMMANDS:
            npos, nfacing, ngmask, nsmask = pos, facing, gmask, smask
            goal = False
            if cmd == "turn_left":
                nfacing = LEFT[facing]
            elif cmd == "turn_right":
                nfacing = RIGHT[facing]
            elif cmd == "move_forward":
                dx, dy = DIR_VEC[facing]
                npos = (pos[0] + dx, pos[1] + dy)
                if not walkable(npos, smask):
                    continue
                goal = npos == level.goal
            elif cmd == "collect_gem":
                if pos in gem_index and (gmask >> gem_index[pos]) & 1:
                    ngmask = gmask & ~(1 << gem_index[pos])
                else:
                    continue
            elif cmd == "toggle_switch":
                hit = False
                for k, sw in enumerate(level.switches):
                    if sw.pos == pos:
                        nsmask ^= (1 << k)
                        hit = True
                if not hit:
                    continue
            nst = (npos, nfacing, ngmask, nsmask)
            nd = d + 1
            if goal:
                collected = len(gem_list) - bin(ngmask).count("1")
                score = SCORE_GOAL + SCORE_GEM * collected + SCORE_PER_COMMAND * nd
                if best is None or score > best[0]:
                    prev[("GOAL", nst, nd)] = (st, cmd)
                    best = (score, ("GOAL", nst, nd))
                continue
            if nd < dist.get(nst, 1 << 30):
                dist[nst] = nd
                prev[nst] = (st, cmd)
                heapq.heappush(pq, (nd, tie, nst))
                tie += 1

    if best is None:
        return None, 0
    # Reconstruct.
    program: list[str] = []
    node = best[1]
    while node in prev:
        node, cmd = prev[node]
        program.append(cmd)
    program.reverse()
    return program, best[0]
