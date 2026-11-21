"""Small current-dispatch attention over appointments and nurse roster rows."""
from __future__ import annotations

from math import sqrt

import torch
from torch import nn
from stable_baselines3.common.torch_layers import CombinedExtractor


class DispatchAttention(CombinedExtractor):
    """Append separate attention summaries to the original flattened features."""

    def __init__(self, observation_space, *, width=16, horizon=480., speed=60.):
        super().__init__(observation_space)
        self._features_dim += 2 * width
        self.scale = sqrt(width)
        self.travel_per_shift = speed * horizon
        self.queries = nn.Linear(12, 2 * width)
        self.jobs = nn.Linear(observation_space['jobs'].shape[-1] + 3, 2 * width)
        self.nurses = nn.Linear(observation_space['nurses'].shape[-1], 2 * width)

    def _pool(self, tokens, query, mask=None):
        keys, values = tokens.tanh().chunk(2, dim=-1)
        scores = (keys * query.unsqueeze(1)).sum(dim=-1) / self.scale
        if mask is None:
            weights = scores.softmax(dim=1)
        else:
            weights = scores.masked_fill(~mask, torch.finfo(scores.dtype).min).softmax(dim=1) * mask
        return (values * weights.unsqueeze(-1)).sum(dim=1)

    def forward(self, obs):
        jobs, worker = obs['jobs'], obs['worker']
        clock = obs['global'][:, :1]
        x = (jobs[..., 0] - worker[:, 0:1]) * 250.
        y = (jobs[..., 1] - worker[:, 1:2]) * 250.
        job_floor, worker_floor = jobs[..., 2] * 30., worker[:, 2:3] * 30.
        horizontal = torch.hypot(x, y)
        distance = torch.where(horizontal == 0, 9.6 * (job_floor - worker_floor).abs(),
                               horizontal + job_floor + worker_floor)
        arrival = clock + distance / self.travel_per_shift
        waiting = (jobs[..., 3] - arrival).clamp_min(0.)
        slack = jobs[..., 4] - arrival - waiting
        job_inputs = torch.cat((jobs, torch.stack((obs['eligible'], waiting, slack), dim=-1)), dim=-1)
        job_query, nurse_query = self.queries(torch.cat((worker, obs['global']), dim=-1)).chunk(2, dim=-1)
        job_context = self._pool(self.jobs(job_inputs), job_query, jobs[..., 6] == 0)
        nurse_context = self._pool(self.nurses(obs['nurses']), nurse_query)
        return torch.cat((super().forward(obs), job_context, nurse_context), dim=-1)
