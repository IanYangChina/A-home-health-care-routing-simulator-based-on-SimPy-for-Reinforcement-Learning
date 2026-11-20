"""All policies act on Gymnasium observations/masks; no hidden future RNG."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def _feasible(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(mask[:-1], dtype=bool))


class RandomPolicy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, observation: dict, info: dict) -> int:
        choices = _feasible(info['action_mask'])
        if not len(choices):
            return len(info['action_mask']) - 1
        return int(self.rng.choice(choices))


class GreedyPolicy:
    """Earliest-window/shortest-travel heuristic; mask enforces constraints."""
    def act(self, observation: dict, info: dict) -> int:
        choices = _feasible(info['action_mask'])
        if not len(choices):
            return len(info['action_mask']) - 1
        jobs, w, now = observation['jobs'], observation['worker'], observation['global'][0]
        # Normalised location and time; scalar weights are heuristic only.
        dist = np.linalg.norm(jobs[choices, :2] - w[:2], axis=1)
        urgency = np.maximum(0., jobs[choices, 4] - now)
        wait = np.maximum(0., jobs[choices, 3] - now - dist * .02)
        score = 3. * dist + .3 * urgency + .7 * wait
        return int(choices[np.argmin(score)])


class EarliestDeadlinePolicy:
    """Choose the feasible appointment with the earliest closing time, then lowest ID."""

    def act(self, observation: dict, info: dict) -> int:
        choices = _feasible(info['action_mask'])
        if not len(choices):
            return len(info['action_mask']) - 1
        return int(choices[np.argmin(observation['jobs'][choices, 4])])


@dataclass
class ACOPolicy:
    """A small ant-colony lookahead planner producing one Gymnasium action.

    Candidate routes are *forecasts* for the currently selected nurse, based
    entirely on observable rows and expected service times. Each `act` returns
    the first job of the best ant's route. Pheromones persist within an episode.

    This is a new event-dispatch ACO baseline, NOT the original QL-BWACO
    algorithm or its chance-constrained optimisation formulation.
    """
    ants: int = 10
    iterations: int = 3
    lookahead: int = 4
    alpha: float = 1.0
    beta: float = 2.0
    evaporation: float = .25
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.trails: np.ndarray | None = None

    def reset(self):
        self.trails = None

    def act(self, observation: dict, info: dict) -> int:
        allowed = _feasible(info['action_mask'])
        n = len(info['action_mask']) - 1
        if self.trails is None or self.trails.shape != (n + 1, n + 1):
            self.trails = np.ones((n + 1, n + 1), dtype=float)
        if not len(allowed):
            return n
        if len(allowed) == 1:
            return int(allowed[0])
        jobs = observation['jobs']
        nurse = observation['worker']
        t0 = float(observation['global'][0])
        best_route: list[int] = []
        best_score = float('-inf')
        # Locations and times are normalized, but relative comparisons still
        # respect windows and route geometry in the observed feature space.
        for _ in range(self.iterations):
            iteration_best = None
            iteration_score = float('-inf')
            for _ in range(self.ants):
                used = set()
                route = []
                score = 0.
                current_pos = nurse[:3].astype(float).copy()
                now = t0
                current_node = n  # depot/sentinel row for pheromone edges
                for _ in range(self.lookahead):
                    feasible = []
                    weights = []
                    preds = []
                    for j in allowed:
                        j = int(j)
                        if j in used:
                            continue
                        job = jobs[j]
                        # Use a conservative approximate travel-time forecast;
                        # actual feasibility is rechecked by the Gym mask.
                        dx = current_pos[0] - float(job[0]); dy = current_pos[1] - float(job[1])
                        dz = current_pos[2] - float(job[2])
                        # 250 units / 60 units/min / 480 min = 0.00868
                        travel = ((dx * dx + dy * dy) ** .5 * 250 + abs(dz) * 30) / (60 * 480)
                        arrive = now + travel
                        wait = max(0., float(job[3]) - arrive)
                        start = arrive + wait
                        end = start + float(job[7])
                        if start > float(job[4]) + 1e-8 or end > 1.:
                            continue
                        feasibility = (travel, wait, end)
                        attractiveness = 1. / (.01 + travel + .45 * wait + .04 * float(job[7]))
                        feasible.append(j)
                        weights.append((self.trails[current_node, j] ** self.alpha)
                                       * attractiveness ** self.beta)
                        preds.append(feasibility)
                    if not feasible:
                        break
                    weights = np.asarray(weights, dtype=float)
                    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)
                    ix = int(self.rng.choice(len(feasible), p=weights))
                    j = feasible[ix]
                    travel, wait, end = preds[ix]
                    route.append(j)
                    used.add(j)
                    current_pos = jobs[j, :3].astype(float).copy()
                    current_node = j
                    now = end
                    score += 1. - .15 * travel - .30 * wait - .02 * float(jobs[j, 7])
                if route and score > iteration_score:
                    iteration_score, iteration_best = score, route
            self.trails *= (1. - self.evaporation)
            if iteration_best:
                prev = n
                for job_id in iteration_best:
                    self.trails[prev, job_id] += .25 + iteration_score / max(1, len(iteration_best))
                    prev = job_id
            if iteration_best and iteration_score > best_score:
                best_score, best_route = iteration_score, iteration_best
        return int(best_route[0]) if best_route else int(allowed[0])
