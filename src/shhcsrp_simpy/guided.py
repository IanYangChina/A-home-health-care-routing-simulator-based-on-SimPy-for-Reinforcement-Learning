"""Dispatch teachers for DQN behavior collection and PPO imitation warm-start."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from .policies import ACOPolicy


def _short_wait_action(env, mask) -> int:
    jobs = np.flatnonzero(mask[:-1]).tolist()
    if not jobs:
        return len(mask) - 1
    sim = env.unwrapped.model
    nurse = sim.current_worker

    def priority(job):
        forecast = sim.predict(nurse, job)
        return forecast['wait'], sim.jobs[job].window_end, forecast['travel'], job

    return min(jobs, key=priority)


def _bounded_aco(seed: int) -> ACOPolicy:
    return ACOPolicy(ants=3, iterations=1, lookahead=3, seed=seed)


class ExplorationGuide:
    """Mix teacher suggestions into DQN's exploratory actions during collection."""

    def __init__(self, env, strategy: str, seed: int):
        if strategy not in ('short_wait', 'aco'):
            raise ValueError(f'Unknown exploration guide {strategy!r}')
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.aco = _bounded_aco(seed) if strategy == 'aco' else None
        self.counts = Counter()
        self.episode = None

    def act(self, observation, uniform_action: int) -> int:
        mask = self.env.action_masks()
        if not mask[:-1].any():
            self.counts['forced_defer'] += 1
            return len(mask) - 1
        draw = self.rng.random()
        if draw < .25:
            self.counts['uniform'] += 1
            return uniform_action
        if draw < .45 and self.aco is not None:
            episode = self.env.unwrapped.model
            if episode is not self.episode:
                self.aco.reset()
                self.episode = episode
            self.counts['aco'] += 1
            return self.aco.act(observation, {'action_mask': mask})
        self.counts['short_wait'] += 1
        return _short_wait_action(self.env, mask)


@dataclass
class Demonstrations:
    observations: dict[str, np.ndarray]
    masks: np.ndarray
    actions: np.ndarray
    episodes: list[dict]


def collect_demonstrations(env_factory, strategy: str, seeds) -> Demonstrations:
    """Collect actual masked teacher decisions with separate scenario seeds."""
    if strategy not in ('short_wait', 'aco'):
        raise ValueError(f'Unknown demonstration teacher {strategy!r}')
    observations = defaultdict(list)
    masks, actions, episodes = [], [], []
    env = env_factory()
    try:
        for seed in seeds:
            aco = _bounded_aco(seed) if strategy == 'aco' else None
            obs, info = env.reset(seed=seed)
            total = 0.
            for step in range(1, 5001):
                mask = info['action_mask']
                action = (aco.act(obs, info) if aco is not None
                          else _short_wait_action(env, mask))
                if mask[:-1].any():
                    for key, values in obs.items():
                        observations[key].append(values.copy())
                    masks.append(mask.copy())
                    actions.append(action)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            else:
                raise RuntimeError(f'Teacher exceeded 5000 decisions on seed {seed}')
            episodes.append({'seed': seed, 'steps': step,
                             'jobs_completed': info['metrics']['jobs_completed'],
                             'total_reward': round(total, 4)})
    finally:
        env.close()
    if not actions:
        raise ValueError('Teacher trajectories contained no eligible visits')
    return Demonstrations({key: np.stack(values) for key, values in observations.items()},
                          np.stack(masks), np.asarray(actions), episodes)


def warm_start_ppo(model, demonstrations: Demonstrations, seed: int, *,
                   epochs: int = 8, batch_size: int = 128, learning_rate: float = .0003) -> dict:
    """Fit teacher actions by masked imitation before ordinary PPO rollout updates."""
    policy = model.policy
    obs, _ = policy.obs_to_tensor(demonstrations.observations)
    actions = torch.as_tensor(demonstrations.actions, device=model.device, dtype=torch.long)
    masks = torch.as_tensor(demonstrations.masks, device=model.device, dtype=torch.bool)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    def batch_loss(indices):
        subset = {key: values[indices] for key, values in obs.items()}
        _, log_prob, _ = policy.evaluate_actions(subset, actions[indices], action_masks=masks[indices])
        return -log_prob.mean()

    def measure():
        with torch.no_grad():
            batches = np.array_split(np.arange(len(actions)),
                                     (len(actions) + batch_size - 1) // batch_size)
            return sum(float(batch_loss(batch)) * len(batch) for batch in batches) / len(actions)

    loss_before = measure()
    policy.set_training_mode(True)
    for _ in range(epochs):
        order = rng.permutation(len(actions))
        for start in range(0, len(actions), batch_size):
            optimizer.zero_grad()
            batch_loss(order[start:start + batch_size]).backward()
            optimizer.step()
    policy.set_training_mode(False)
    return {'examples': len(actions), 'steps': sum(ep['steps'] for ep in demonstrations.episodes),
            'episodes': demonstrations.episodes, 'epochs': epochs, 'batch_size': batch_size,
            'learning_rate': learning_rate, 'loss_before': loss_before, 'loss_after': measure()}
