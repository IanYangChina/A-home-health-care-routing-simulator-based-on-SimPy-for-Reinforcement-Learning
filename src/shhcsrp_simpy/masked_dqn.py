"""Stable-Baselines3 DQN with appointment eligibility masks."""
from __future__ import annotations

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MultiInputPolicy, QNetwork


class MaskedQNetwork(QNetwork):
    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        values = super().forward(obs)
        eligible = obs['eligible'].bool()
        defer = torch.ones_like(eligible[:, :1])
        return values.masked_fill(~torch.cat((eligible, defer), dim=1), -torch.inf)


class MultiInputQPolicy(MultiInputPolicy):
    def make_q_net(self) -> QNetwork:
        # SB3 builds the online and target networks with this same factory.
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return MaskedQNetwork(**net_args).to(self.device)


class MaskedDQN(DQN):
    """Apply eligibility in replay targets, greedy prediction, and exploration."""

    policy_aliases = {'MultiInputPolicy': MultiInputQPolicy}

    def _sample_eligible(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        eligible = observation['eligible']
        rows = np.atleast_2d(eligible)
        masks = np.ones((len(rows), self.action_space.n), dtype=np.int8)
        masks[:, :-1] = rows
        actions = np.array([self.action_space.sample(mask=mask) for mask in masks])
        return actions if eligible.ndim == 2 else actions.squeeze(axis=0)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if not deterministic and np.random.rand() < self.exploration_rate:
            return self._sample_eligible(observation), state
        return self.policy.predict(observation, state, episode_start, deterministic)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        guide = getattr(self, 'exploration_guide', None)
        if guide is not None:
            if n_envs != 1:
                raise ValueError('Guided exploration requires one live environment')
            if self.num_timesteps < learning_starts or np.random.rand() < self.exploration_rate:
                uniform = int(self._sample_eligible(self._last_obs)[0])
                observation = {key: values[0] for key, values in self._last_obs.items()}
                actions = np.array([guide.act(observation, uniform)])
            else:
                actions, _ = self.predict(self._last_obs, deterministic=True)
            return actions, actions
        if self.num_timesteps < learning_starts:
            # SB3 samples directly from the action space during replay warm-up.
            actions = self._sample_eligible(self._last_obs)
            return actions, actions
        return super()._sample_action(learning_starts, action_noise, n_envs)

    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), 'exploration_guide']
