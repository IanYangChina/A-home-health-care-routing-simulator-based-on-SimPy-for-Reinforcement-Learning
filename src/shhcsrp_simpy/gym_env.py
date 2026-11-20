"""Gymnasium facade over a real SimPy discrete-event execution engine."""
from __future__ import annotations

from math import isfinite
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .data import Scenario, load_scenario
from .simulation import CareSimulation


class SHHCSRPGymEnv(gym.Env):
    """Dispatch the next job for the current idle nurse, then advance time.

    Action: Discrete(n_jobs + 1), where n_jobs = defer this nurse.
    Decision epochs: idle nurse ready; other nurses continue concurrently.
    Observations: padded-free fixed arrays for one selected dataset instance.
    Use `MaskablePPO` from sb3-contrib if respecting feasibility masks.
    """
    metadata = {'render_modes': ['ansi'], 'render_fps': 4}

    def __init__(self, *, scenario: Scenario | None = None, instance: str = 'A',
                 data_dir=None, job_limit: int | None = None,
                 duration_cv: float = .15, render_mode: str | None = None,
                 common_random_numbers: bool = False, wait_penalty: float = .05,
                 terminal_shift: bool = False):
        super().__init__()
        self.scenario = scenario if scenario is not None else load_scenario(
            instance=instance, root=data_dir, job_limit=job_limit)
        self.duration_cv = float(duration_cv)
        if not 0 <= self.duration_cv <= 1.:
            raise ValueError('duration_cv must be in [0,1]')
        self.common_random_numbers = common_random_numbers
        self.wait_penalty = float(wait_penalty)
        if not isfinite(self.wait_penalty) or self.wait_penalty < 0.:
            raise ValueError('wait_penalty must be finite and nonnegative')
        self.terminal_shift = terminal_shift
        self.n_jobs = len(self.scenario.jobs)
        max_elder_id = max(job.elder_id for job in self.scenario.jobs)
        horizon = self.scenario.horizon
        self._job_features = np.asarray([
            (job.position[0] / 250., job.position[1] / 250., job.position[2] / 30.,
             job.window_start / horizon, job.window_end / horizon, job.demand / 3.,
             0., 0., job.elder_id / max_elder_id)
            for job in self.scenario.jobs
        ], dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_jobs + 1)
        self.observation_space = spaces.Dict({
            'jobs': spaces.Box(0., 1., shape=(self.n_jobs, 9), dtype=np.float32),
            'worker': spaces.Box(0., 1., shape=(8,), dtype=np.float32),
            'global': spaces.Box(0., 1., shape=(4,), dtype=np.float32),
        })
        if render_mode not in (None, 'ansi'):
            raise ValueError("Only render_mode='ansi' supported; use export_html for plots")
        self.render_mode = render_mode
        self.model: CareSimulation | None = None
        self._seed = 0
        self._episode_steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._seed = int(self.np_random.integers(0, 2**31 - 1))
        self.model = CareSimulation(self.scenario, seed=self._seed,
                                    duration_cv=self.duration_cv,
                                    common_random_numbers=self.common_random_numbers,
                                    wait_penalty=self.wait_penalty)
        self._episode_steps = 0
        self.model.advance()
        return self._observe(), self._info()

    def action_masks(self) -> np.ndarray:
        if self.model is None:
            return np.array([False] * self.n_jobs + [True], dtype=bool)
        return self.model.action_masks()

    def step(self, action: int):
        if self.model is None:
            raise RuntimeError('Call reset() before step().')
        if self.model.done:
            raise RuntimeError('Episode has ended; call reset().')
        m = self.model
        a = int(action)
        valid = a == self.n_jobs or m.dispatch(a)
        if a == self.n_jobs or not valid:
            m.defer()
            if not valid:
                m.reward_buffer -= 3.
                m._log('invalid_action', detail=str(a))
        self._episode_steps += 1
        m.advance()
        reward = float(m.reward_buffer)
        m.reward_buffer = 0.
        info = self._info()
        info['invalid_action'] = not valid
        truncated = bool(m.truncated and not self.terminal_shift)
        return self._observe(), reward, bool(m.done and not truncated), truncated, info

    def _observe(self) -> dict[str, np.ndarray]:
        assert self.model is not None
        m = self.model
        H = self.scenario.horizon
        jid = m.current_worker
        obs_jobs = self._job_features.copy()
        obs_jobs[:, 6] = m.status / 3.
        worker = np.zeros(8, dtype=np.float32)
        if jid is not None:
            obs_jobs[:, 7] = m.expected_durations(jid) / H
            w = m.workers[jid]
            worker[:] = (w.position[0] / 250., w.position[1] / 250.,
                         w.position[2] / 30., w.skill / 3.,
                         jid / max(1, len(m.workers) - 1),
                         m.env.now / H, w.completed / self.n_jobs,
                         (H - float(m.env.now)) / H)
        globals_ = np.asarray([
            float(m.env.now) / H,
            float(np.sum(m.status == 2)) / self.n_jobs,
            float(np.sum(m.status == 0)) / self.n_jobs,
            float(m.active) / len(m.workers),
        ], dtype=np.float32)
        # Bounds stay valid even if a user's custom scenario extends beyond
        # original x/y/z ranges or a stochastic event runs beyond its shift.
        return {'jobs': np.clip(obs_jobs, 0., 1.), 'worker': np.clip(worker, 0., 1.),
                'global': np.clip(globals_, 0., 1.)}

    def _info(self) -> dict:
        assert self.model is not None
        return {'action_mask': self.model.action_masks().copy(),
                'current_nurse': self.model.current_worker,
                'sim_time': float(self.model.env.now),
                'metrics': self.model.metrics() if self.model.done else None}

    def render(self):
        if self.model is None:
            return 'Not started'
        m = self.model
        return (f't={m.env.now:.1f} min | nurse={m.current_worker} | '
                f'completed={np.sum(m.status == 2)}/{self.n_jobs} | '
                f'active={m.active} | pending={len(m.pending)}')

    def close(self):
        self.model = None
