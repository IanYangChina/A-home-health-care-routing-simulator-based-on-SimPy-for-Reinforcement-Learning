"""Masked n-step DQN for dictionary observations in Stable-Baselines3 2.6."""
from __future__ import annotations

from collections import deque
from typing import NamedTuple

import numpy as np
import torch
from stable_baselines3.common.buffers import DictReplayBuffer
from torch.nn import functional as F

from .masked_dqn import MaskedDQN


class NStepSamples(NamedTuple):
    observations: dict[str, torch.Tensor]
    actions: torch.Tensor
    next_observations: dict[str, torch.Tensor]
    dones: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor


class NStepDictReplayBuffer(DictReplayBuffer):
    """Store one discounted return per transition, including shorter episode tails."""

    def __init__(self, *args, n_steps, gamma, **kwargs):
        super().__init__(*args, **kwargs)
        if self.n_envs != 1:
            raise ValueError('N-step replay requires exactly one environment')
        if n_steps < 1:
            raise ValueError('n_steps must be positive')
        self.n_steps = n_steps
        self._powers = [gamma ** index for index in range(n_steps + 1)]
        self.discounts = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self._pending = deque()

    def reset(self):
        super().reset()
        self._pending.clear()

    def add(self, obs, next_obs, action, reward, done, infos):
        self._pending.append(({key: value.copy() for key, value in obs.items()},
                              action.copy(), float(reward[0])))
        # SB3 supplies the real final observation even when its vector env autoresets.
        if done[0]:
            while self._pending:
                self._emit(next_obs, done, infos)
        elif len(self._pending) == self.n_steps:
            self._emit(next_obs, done, infos)

    def _emit(self, next_obs, done, infos):
        span = len(self._pending)
        reward = sum(power * transition[2] for power, transition in zip(self._powers, self._pending))
        obs, action, _ = self._pending.popleft()
        self.discounts[self.pos, 0] = self._powers[span]
        super().add(obs, next_obs, action, np.array([reward], dtype=np.float32), done, infos)

    def _get_samples(self, batch_inds, env=None):
        batch = super()._get_samples(batch_inds, env)
        return NStepSamples(batch.observations, batch.actions, batch.next_observations,
                            batch.dones, batch.rewards, self.to_torch(self.discounts[batch_inds]))


class NStepMaskedDQN(MaskedDQN):
    """Use the existing action masks and an n-step bootstrapped DQN target."""

    def __init__(self, policy, env, *, n_steps=16, _init_setup_model=True, **kwargs):
        if n_steps < 1:
            raise ValueError('n_steps must be positive')
        super().__init__(policy, env, replay_buffer_class=NStepDictReplayBuffer,
                         _init_setup_model=False, **kwargs)
        self.return_steps = n_steps
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self):
        self.replay_buffer_kwargs = {**self.replay_buffer_kwargs,
                                     'n_steps': self.return_steps, 'gamma': self.gamma}
        super()._setup_model()

    def _setup_learn(self, total_timesteps, callback=None, reset_num_timesteps=True,
                     tb_log_name='run', progress_bar=False):
        # Discard unresolved prefixes when starting a fresh environment episode.
        if reset_num_timesteps or self._last_obs is None:
            self.replay_buffer._pending.clear()
        return super()._setup_learn(total_timesteps, callback, reset_num_timesteps,
                                    tb_log_name, progress_bar)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        if not self.replay_buffer.size():
            return
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            batch = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            with torch.no_grad():
                next_q = self.q_net_target(batch.next_observations).max(dim=1, keepdim=True).values
                targets = batch.rewards + (1 - batch.dones) * batch.discounts * next_q
            current = self.q_net(batch.observations).gather(1, batch.actions.long())
            loss = F.smooth_l1_loss(current, targets)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')
        self.logger.record('train/loss', np.mean(losses))
