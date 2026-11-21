"""Tune SB3 dispatch agents and compare them with four fixed policies."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
from importlib.metadata import version
import json
from math import isfinite
from pathlib import Path
from shutil import copyfile
from statistics import mean, stdev
from time import perf_counter
from weakref import proxy

import gymnasium as gym
import numpy as np
import optuna
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger

from .data import Scenario, load_scenario
from .gym_env import SHHCSRPGymEnv
from .masked_dqn import MaskedDQN
from .policies import ACOPolicy, EarliestDeadlinePolicy, GreedyPolicy, RandomPolicy
from .search import WIDTHS, suggest_params, training_settings


VALIDATION_SEEDS = (51, 52, 53, 54, 55)
TEST_SEEDS = tuple(range(101, 113))
PROFILE_INTERVAL = 512
ALGORITHMS = {'dqn': MaskedDQN, 'ppo': MaskablePPO}
METHODS = [
    ('random', 'Random', 'baseline', 'Uniform sample from eligible appointments.'),
    ('greedy', 'Greedy', 'baseline', 'Scores eligible appointments by XY distance, closing time and waiting.'),
    ('edf', 'Earliest deadline', 'baseline', 'Chooses the earliest closing window, breaking ties by appointment index.'),
    ('aco', 'Ant colony', 'baseline', 'Samples short sequences of eligible appointments and takes the first visit of the best forecast.'),
    ('dqn', 'DQN', 'rl', 'Learns Q-values for fixed appointment IDs and defer. Eligibility masks apply during exploration, action selection and target calculation.'),
    ('ppo', 'PPO', 'rl', 'MaskablePPO learns a policy over fixed appointment IDs and defer. It samples and selects only eligible actions.'),
]


class StableDispatch(gym.Wrapper):
    """Add eligibility observations while preserving the original job-ID actions."""

    def __init__(self, env: SHHCSRPGymEnv):
        super().__init__(env)
        self.n_jobs = env.n_jobs
        self.observation_space = gym.spaces.Dict({
            **env.observation_space.spaces,
            'eligible': gym.spaces.Box(0., 1., (self.n_jobs,), dtype=np.float32),
        })
        self._mask = env.action_masks()

    def _observe(self, obs, info):
        self._mask = info['action_mask'].copy()
        return {**obs, 'eligible': self._mask[:-1].astype(np.float32)}

    def action_masks(self):
        return self._mask.copy()

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._observe(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._observe(obs, info), reward, terminated, truncated, info


class TrainingHistory(Logger):
    """Record SB3 loss samples at their update step, independently of log dumps."""

    metrics = ('train/loss', 'train/policy_gradient_loss', 'train/value_loss',
               'train/entropy_loss', 'train/approx_kl', 'rollout/ep_rew_mean')

    def __init__(self, model):
        super().__init__(folder=None, output_formats=[])
        self.model = proxy(model)
        self.records = {}

    def record(self, key, value, exclude=None):
        super().record(key, value, exclude)
        if key in self.metrics and isfinite(value):
            step = self.model.num_timesteps
            self.records.setdefault(step, {})[key] = float(value)


class EpisodeCounter(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episodes = 0

    def _on_step(self):
        self.episodes += int(np.count_nonzero(self.locals['dones']))
        return True


def loss_profiles(histories, steps):
    """Give each training seed equal weight within each environment-step window."""
    across_seeds = defaultdict(list)
    for history in histories:
        within_seed = defaultdict(list)
        for step, metrics in history.items():
            endpoint = min(steps, ((max(1, step) - 1) // PROFILE_INTERVAL + 1) * PROFILE_INTERVAL)
            for key, value in metrics.items():
                within_seed[key, endpoint].append(value)
        for key, values in within_seed.items():
            across_seeds[key].append(mean(values))
    profiles = defaultdict(list)
    for (key, step), values in sorted(across_seeds.items()):
        profiles[key].append({'steps': step, 'mean': mean(values), 'n': len(values),
                              'std': stdev(values) if len(values) > 1 else None})
    return dict(profiles)


def make_env(scenario: Scenario, method: str, **env_options):
    env = SHHCSRPGymEnv(scenario=scenario, common_random_numbers=True, **env_options)
    if method in ALGORITHMS:
        return StableDispatch(env)
    return env


def evaluate(scenario, method, seed, *, model=None, trace=False, nurse_roster=False):
    start = perf_counter()
    env = make_env(scenario, method, nurse_roster=nurse_roster)
    if model is None:
        agent = {
            'random': lambda: RandomPolicy(seed=seed),
            'greedy': GreedyPolicy,
            'edf': EarliestDeadlinePolicy,
            'aco': lambda: ACOPolicy(seed=seed),
        }[method]()
    obs, info = env.reset(seed=seed)
    total_reward = 0.
    decisions = 0
    decision_limit = max(5000, (2 * len(scenario.jobs) + 1) * len(scenario.nurse_skills))
    try:
        for decisions in range(1, decision_limit + 1):
            if model is None:
                action = agent.act(obs, info)
            elif method == 'ppo':
                action = model.predict(obs, deterministic=True, action_masks=info['action_mask'])[0]
            else:
                action = model.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        else:
            raise RuntimeError(f'{method} exceeded {decision_limit} decisions on seed {seed}')
        metrics = {**info['metrics'], 'total_reward': round(total_reward, 4),
                   'decisions': decisions, 'wall_seconds': round(perf_counter() - start, 5)}
        result = {'method': method, 'seed': seed, 'metrics': metrics}
        if trace:
            result['events'] = env.unwrapped.model.events
        return result
    finally:
        env.close()


def train(method, scenario, params, *, device, seed, steps, network=None, record_history=False):
    settings = training_settings(method, params, steps, network=network or (64, 64))
    env = make_env(scenario, method)
    try:
        model = ALGORITHMS[method]('MultiInputPolicy', env, **settings,
                                   device=device, seed=seed, verbose=0)
        history = counter = None
        if record_history:
            history = TrainingHistory(model)
            counter = EpisodeCounter()
            model.set_logger(history)
        model.learn(total_timesteps=steps, callback=counter, log_interval=1 if record_history else None)
        return model, history.records if history is not None else {}, counter.episodes if counter else 0
    finally:
        env.close()


def tune(method, scenario, args, device, reference=None):
    def objective(trial):
        start = perf_counter()
        params = suggest_params(trial, method, tune_width=args.network is None)
        model, _, _ = train(method, scenario, params, device=device, seed=args.training_seed,
                            steps=args.trial_steps, network=args.network)
        rewards = [evaluate(scenario, method, seed, model=model)['metrics']['total_reward']
                   for seed in VALIDATION_SEEDS]
        trial.set_user_attr('steps', model.num_timesteps)
        trial.set_user_attr('seconds', round(perf_counter() - start, 3))
        return mean(rewards)

    trials = []
    if reference is None:
        sampler = optuna.samplers.TPESampler(seed=args.training_seed, n_startup_trials=4)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=args.trials)
        best_params, best_value = study.best_params, study.best_value
        trials = [{'number': t.number, 'value': t.value, 'params': t.params, **t.user_attrs}
                  for t in study.trials]
    else:
        best_params, best_value = dict(reference['best_params']), None
        if args.network is not None:
            best_params.pop('width', None)
    replicates = []
    histories = []
    for seed in args.training_seeds:
        start = perf_counter()
        print(f'Training {method.upper()} with seed {seed}...', flush=True)
        model, history, episodes = train(method, scenario, best_params, device=device, seed=seed,
                                         steps=args.final_steps, network=args.network, record_history=True)
        training_seconds = round(perf_counter() - start, 3)
        validation = mean(evaluate(scenario, method, s, model=model)['metrics']['total_reward']
                          for s in VALIDATION_SEEDS)
        model.save(args.output_dir / f'{method}-{seed}')
        replicates.append({'seed': seed, 'steps': model.num_timesteps, 'episodes': episodes,
                           'seconds': training_seconds, 'validation_reward': validation})
        histories.append(history)
        print(f'{method.upper()} seed {seed}: {episodes} training episodes, validation reward {validation:.2f}',
              flush=True)
        del model
    representative = max(replicates, key=lambda run: run['validation_reward'])
    path = args.output_dir / f'{method}.zip'
    copyfile(args.output_dir / f"{method}-{representative['seed']}.zip", path)
    tuning = {
        'best_params': best_params, 'best_value': best_value, 'trials': trials,
        'final_training_seconds': round(sum(run['seconds'] for run in replicates), 3),
        'final_training_steps': representative['steps'],
        'final_validation_reward': mean(run['validation_reward'] for run in replicates),
        'representative_seed': representative['seed'], 'replicates': replicates,
        'loss_profiles': loss_profiles(histories, representative['steps']),
    }
    return ALGORITHMS[method].load(path, device=device), tuning


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance', choices=list('ABCD'), default='A')
    parser.add_argument('--limit', type=int, default=0, help='first N jobs, or 0 for the full scenario')
    parser.add_argument('--trials', type=int, default=8)
    parser.add_argument('--trial-steps', type=int, default=4096)
    parser.add_argument('--final-steps', type=int, default=24576)
    parser.add_argument('--network', type=int, nargs='+', metavar='WIDTH',
                        help='fix hidden layer sizes for both agents')
    parser.add_argument('--fixed-params', type=Path, metavar='BENCHMARK_JSON',
                        help='reuse tuned parameters from an earlier run of the same scenario')
    parser.add_argument('--training-seed', type=int, default=11)
    parser.add_argument('--replicates', type=int, default=3,
                        help='independent final training runs per agent, starting at --training-seed')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--ppo-device', choices=['auto', 'cuda', 'cpu'], help='override PPO training device')
    parser.add_argument('--output-dir', type=Path, default=Path('output/benchmark'))
    parser.add_argument('--report', type=Path, help='HTML path, defaults to report.html in the output directory')
    parser.add_argument('--report-only', type=Path, metavar='BENCHMARK_JSON')
    args = parser.parse_args(argv)
    args.report = args.report or args.output_dir / 'report.html'
    if args.report_only is not None:
        from .benchmark_report import render_report
        print(render_report(json.loads(args.report_only.read_text(encoding='utf-8')), args.report))
        return
    if args.limit < 0 or min(args.trials, args.trial_steps, args.final_steps, args.replicates) < 1:
        parser.error('The limit must be nonnegative and training budgets must be positive')
    if args.network is not None and min(args.network) < 1:
        parser.error('Every hidden layer width must be positive')
    args.training_seeds = tuple(range(args.training_seed, args.training_seed + args.replicates))
    if args.training_seed < 0 or set(args.training_seeds).intersection(VALIDATION_SEEDS + TEST_SEEDS):
        parser.error('Training seeds must be nonnegative and separate from validation and test seeds')
    has_cuda = torch.cuda.is_available()
    if 'cuda' in (args.device, args.ppo_device) and not has_cuda:
        parser.error('CUDA was requested but PyTorch cannot access a CUDA GPU')
    devices = {method: 'cuda' if requested != 'cpu' and has_cuda else 'cpu'
               for method, requested in (('dqn', args.device), ('ppo', args.ppo_device or args.device))}
    device_label = ', '.join(f'{method.upper()} on {device.upper()}' for method, device in devices.items())
    gpu = torch.cuda.get_device_name(0) if has_cuda else 'Unavailable'
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    scenario = load_scenario(args.instance, job_limit=args.limit or None)
    reference = parameters_from = None
    if args.fixed_params is not None:
        source = json.loads(args.fixed_params.read_text(encoding='utf-8'))
        source_meta = source['meta']
        if (source_meta['instance'], source_meta['jobs']) != (scenario.name, len(scenario.jobs)):
            parser.error('The parameter source must use the same instance and number of appointments')
        reference = source['tuning']
        parameters_from = f'instance {scenario.name} study'
        if source_meta.get('network'):
            parameters_from += ' · ' + ' × '.join(map(str, source_meta['network']))
    print(f'{scenario.name}: {len(scenario.jobs)} jobs, {device_label}, '
          f'training seeds {args.training_seeds}', flush=True)
    models = {}
    tuning = {}
    for method in ALGORITHMS:
        print(f'{"Training" if reference else "Tuning"} {method.upper()}...', flush=True)
        model, tuning[method] = tune(method, scenario, args, devices[method],
                                     reference[method] if reference else None)
        models[method] = model
        score = tuning[method]['final_validation_reward']
        print(f'{method.upper()} final mean validation reward: {score:.2f}', flush=True)
    selected = max(tuning, key=lambda method: tuning[method]['final_validation_reward'])
    print(f'Selected {selected.upper()} before held-out evaluation.', flush=True)

    runs = []
    for method, label, _, _ in METHODS:
        outcomes = [evaluate(scenario, method, seed, model=models.get(method), trace=True)
                    for seed in TEST_SEEDS]
        runs.extend(outcomes)
        jobs = mean(run['metrics']['jobs_completed'] for run in outcomes)
        reward = mean(run['metrics']['total_reward'] for run in outcomes)
        print(f'{label}: {jobs:.2f} completed, mean reward {reward:.2f}', flush=True)

    payload = {
        'meta': {
            'instance': scenario.name, 'jobs': len(scenario.jobs), 'nurses': len(scenario.nurse_skills),
            'horizon': scenario.horizon, 'validation_seeds': VALIDATION_SEEDS, 'test_seeds': TEST_SEEDS,
            'training_seed': args.training_seed, 'training_seeds': args.training_seeds,
            'network': args.network, 'network_search': WIDTHS if args.network is None and reference is None else None,
            'parameters_from': parameters_from,
            'trials': 0 if reference else args.trials,
            'trial_steps': 0 if reference else args.trial_steps,
            'final_steps': args.final_steps, 'device': device_label, 'devices': devices, 'gpu': gpu,
            'cuda': torch.version.cuda, 'versions': {pkg: version(pkg) for pkg in
                         ('stable-baselines3', 'sb3-contrib', 'optuna', 'torch', 'gymnasium', 'simpy')},
            'action_encoding': 'Action j keeps appointment ID j throughout a shift. The final action defers. Both agents mask ineligible actions.',
            'observation_normalization': 'Coordinates, times, identifiers and counts use fixed scaling into [0,1]. Eligibility flags are binary. Training uses the original rewards.',
            'service_randomness': 'Each episode seed gives each appointment the same random disturbance across policies. Its duration still depends on the assigned nurse and their prior visits.',
            'loss_profile_note': f'Loss samples are averaged within each {PROFILE_INTERVAL}-step window per training seed. The chart shows their mean and sample standard deviation across independent training seeds.',
            'selection': (
                f'Each algorithm is trained from scratch using the parameters selected in {parameters_from}. '
                if reference else 'Each algorithm is tuned on mean validation reward, then retrained from scratch. '
            ) + 'Mean final validation reward across training seeds selects the RL algorithm. The highest-validation checkpoint for each algorithm supplies the held-out episode traces.',
            'best_rl': selected, 'runtime_seconds': round(perf_counter() - start, 3),
        },
        'scenario': {'depot': scenario.depot, 'nurse_skills': scenario.nurse_skills,
                     'jobs': [asdict(job) for job in scenario.jobs]},
        'methods': [dict(id=method, label=label, group=group, description=description)
                    for method, label, group, description in METHODS],
        'tuning': tuning, 'runs': runs,
    }
    raw_path = args.output_dir / 'benchmark.json'
    raw_path.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    rows = [{'method': run['method'], 'seed': run['seed'], **run['metrics']} for run in runs]
    with (args.output_dir / 'episodes.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {raw_path}', flush=True)
    from .benchmark_report import render_report
    print(render_report(payload, args.report))


if __name__ == '__main__':
    main()
