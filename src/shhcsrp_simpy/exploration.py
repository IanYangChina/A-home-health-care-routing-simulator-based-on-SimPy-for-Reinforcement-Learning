"""Compare guided exploration under the benchmark's original dispatch reward."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
from stable_baselines3.common.callbacks import BaseCallback

from .benchmark import (ALGORITHMS, PROFILE_INTERVAL, TEST_SEEDS, TrainingHistory,
                        evaluate, loss_profiles, make_env)
from .data import load_scenario
from .guided import ExplorationGuide, collect_demonstrations, warm_start_ppo
from .search import training_settings


TRAINING_SEEDS = (11, 12, 13)
DEMO_SEEDS = (201, 202, 203, 204)
ROLLING_EPISODES = 20
REWARD_WEIGHTS = {'completed': 10., 'unserved': -8., 'travel_minutes': -.03,
                  'waiting_minutes': -.05}
VARIANTS = {
    'dqn': (
        ('dqn_uniform', 'DQN uniform', None, 'Uniform legal actions during epsilon exploration.'),
        ('dqn_short_wait', 'DQN short-wait guide', 'short_wait',
         'Epsilon actions use 75% short-wait and 25% uniform exploration when visits are feasible.'),
        ('dqn_aco', 'DQN ACO mix', 'aco',
         'Epsilon actions use 20% bounded ACO, 55% short-wait, and 25% uniform exploration when visits are feasible.'),
    ),
    'ppo': (
        ('ppo_scratch', 'PPO from scratch', None, 'Standard stochastic on-policy rollouts.'),
        ('ppo_short_wait', 'PPO short-wait warm start', 'short_wait',
         'Imitate short-wait demonstrations before collecting ordinary on-policy rollouts.'),
        ('ppo_aco', 'PPO ACO warm start', 'aco',
         'Imitate bounded ACO demonstrations before collecting ordinary on-policy rollouts.'),
    ),
}


class RewardTerms(BaseCallback):
    """Record actual environment totals from each finished training episode."""

    def __init__(self, weights=None):
        super().__init__()
        self.episodes = []
        self.weights = REWARD_WEIGHTS if weights is None else weights

    def _on_step(self):
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            if not done:
                continue
            metrics = info['metrics']
            raw = {'completed': metrics['jobs_completed'], 'unserved': metrics['jobs_unserved'],
                   'travel_minutes': metrics['travel_minutes'], 'waiting_minutes': metrics['waiting_minutes']}
            weighted = {f"{key.removesuffix('_minutes')}_weighted": value * self.weights[key]
                        for key, value in raw.items()}
            predicted = sum(weighted.values())
            actual = info['episode']['r']
            self.episodes.append({'steps': self.num_timesteps, **raw, **weighted,
                                  'reward': predicted, 'realized_reward': actual,
                                  'reward_adjustment': actual - predicted})
            if self.weights is not REWARD_WEIGHTS:
                self.episodes[-1]['reference_reward'] = sum(raw[key] * weight
                                                            for key, weight in REWARD_WEIGHTS.items())
        return True


def rolling_terms(episodes, steps):
    """Sample each seed's trailing completed episodes on a common clock."""
    rolling = deque(maxlen=ROLLING_EPISODES)
    records = {}
    index = 0
    for boundary in range(PROFILE_INTERVAL, steps + PROFILE_INTERVAL, PROFILE_INTERVAL):
        end = min(boundary, steps)
        while index < len(episodes) and episodes[index]['steps'] <= end:
            rolling.append(episodes[index])
            index += 1
        if rolling:
            records[end] = {key: mean(row[key] for row in rolling) for key in rolling[0]
                            if key not in ('steps', 'reward_adjustment')}
    return records


def train_variant(scenario, family, strategy, params, seed, steps, device, demos, *,
                  return_steps=1, weights=None, nurse_roster=False, attention=False):
    algorithm = ALGORITHMS[family]
    env_options = {'nurse_roster': nurse_roster}
    settings = training_settings(family, params, steps)
    if attention:
        from .attention import DispatchAttention

        settings['policy_kwargs'].update(features_extractor_class=DispatchAttention,
                                         features_extractor_kwargs={'horizon': scenario.horizon,
                                                                    'speed': scenario.speed})
    if weights is not None:
        env_options.update(terminal_shift=True, wait_penalty=-weights['waiting_minutes'])
    if return_steps > 1:
        from .nstep_dqn import NStepMaskedDQN

        algorithm = NStepMaskedDQN
        settings['n_steps'] = return_steps
    env = make_env(scenario, family, **env_options)
    try:
        model = algorithm('MultiInputPolicy', env, device=device, seed=seed, verbose=0, **settings)
        guide = demo_stats = None
        if strategy and family == 'dqn':
            guide = ExplorationGuide(env, strategy, seed)
            model.exploration_guide = guide
        elif demos is not None:
            demo_stats = warm_start_ppo(model, demos, seed)
        history = TrainingHistory(model)
        terms = RewardTerms(weights)
        model.set_logger(history)
        model.learn(total_timesteps=steps, callback=terms, log_interval=1)
        return model, history.records, terms.episodes, dict(guide.counts) if guide else {}, demo_stats
    finally:
        env.close()


def run_family(instance, family, steps, device, output_dir, *, configure=None, reuse_dir=None, resume=False):
    start = datetime.now(timezone.utc).isoformat(timespec='seconds')
    scenario = load_scenario(instance)
    prior = Path('output') / f'tuned-{instance.lower()}' / 'benchmark.json'
    source = json.loads(prior.read_text(encoding='utf-8'))
    params = source['tuning'][family]['best_params']
    variants = VARIANTS[family]
    variant_settings = {}
    if configure is not None:
        variants, variant_settings = configure(instance, family, params)
    folder = output_dir / instance.lower()
    folder.mkdir(parents=True, exist_ok=True)
    print(f'{instance} {family.upper()} on {device}, {steps} steps per seed, tuned settings {params}', flush=True)
    methods, experiments, routes, training_episodes = [], {}, [], {}
    demonstrations = {}
    archived = None
    if reuse_dir is not None and family == 'dqn':
        archive = reuse_dir / instance.lower() / 'dqn.json'
        archived = json.loads(archive.read_text(encoding='utf-8'))
        protocol = (instance, steps, list(TRAINING_SEEDS), list(TEST_SEEDS))
        keys = ('instance', 'steps', 'training_seeds', 'test_seeds')
        if tuple(archived['meta'][key] for key in keys) != protocol:
            raise ValueError(f'Mismatched archived DQN protocol for {instance}')
    meta = {'instance': instance, 'family': family, 'device': device, 'steps': steps,
            'training_seeds': list(TRAINING_SEEDS), 'test_seeds': list(TEST_SEEDS),
            'params': params, 'started_at': start}
    if configure is not None:
        meta['variant_settings'] = variant_settings
    if archived is not None:
        meta['reused_dir'] = reuse_dir.as_posix()
    family_result = {'meta': meta, 'methods': methods, 'experiments': experiments, 'routes': routes,
                     'training_episodes': training_episodes}
    partial = folder / f'{family}.partial.json'
    if resume and partial.exists():
        saved = json.loads(partial.read_text(encoding='utf-8'))
        if any(saved['meta'].get(k) != v for k, v in meta.items() if k != 'started_at'):
            raise ValueError(f'Mismatched checkpointed experiment protocol for {instance} {family}')
        meta['started_at'] = saved['meta']['started_at']
        experiments.update(saved['experiments'])
        routes.extend(saved['routes'])
        training_episodes.update(saved['training_episodes'])
    for variant, label, strategy, description in variants:
        methods.append({'id': variant, 'label': label, 'family': family, 'description': description})
        if variant in experiments:
            continue
        changes = variant_settings.get(variant, {})
        weights = changes.get('weights')
        learning_params = changes.get('params', params)
        return_steps = changes.get('return_steps', 1)
        nurse_roster = changes.get('nurse_roster', False)
        attention = changes.get('attention', False)
        archived_id = variant.removesuffix('_mlp')
        if archived and archived_id in archived['experiments']:
            if {key: value for key, value in changes.items() if key != 'attention'} != archived['meta']['variant_settings'][archived_id]:
                raise ValueError(f'Mismatched archived DQN settings for {instance} {variant}')
            experiments[variant] = archived['experiments'][archived_id]
            training_episodes[variant] = archived['training_episodes'][archived_id]
            routes.extend({**route, 'method': variant} for route in archived['routes'] if route['method'] == archived_id)
            print(f'{label}: reused three matched Round 9 fits', flush=True)
            continue
        demos = None
        if family == 'ppo' and strategy:
            demo_options = ({'terminal_shift': True, 'wait_penalty': -weights['waiting_minutes']}
                            if weights is not None else {})
            demo_options['nurse_roster'] = nurse_roster
            key = (strategy, tuple(demo_options.items()))
            if key not in demonstrations:
                demonstrations[key] = collect_demonstrations(
                    lambda: make_env(scenario, family, **demo_options), strategy, DEMO_SEEDS)
            demos = demonstrations[key]
        replications, losses, terms, episodes = [], [], [], []
        training_episodes[variant] = []
        for seed in TRAINING_SEEDS:
            begun = perf_counter()
            print(f'Training {label}, seed {seed}', flush=True)
            model, losses_by_step, recorded, sources, demo_stats = train_variant(
                scenario, family, strategy, learning_params, seed, steps, device, demos,
                return_steps=return_steps, weights=weights, nurse_roster=nurse_roster, attention=attention)
            actual_steps = model.num_timesteps
            model.save(folder / f'{variant}-{seed}')
            summary = {'seed': seed, 'steps': actual_steps, 'episodes': len(recorded),
                       'seconds': round(perf_counter() - begun, 3), 'guide_counts': sources,
                       'demo_examples': demo_stats['examples'] if demo_stats else 0,
                       'demo_stats': demo_stats}
            replications.append(summary)
            training_episodes[variant].append({'seed': seed, 'episodes': recorded})
            losses.append(losses_by_step)
            terms.append(rolling_terms(recorded, actual_steps))
            for test_seed in TEST_SEEDS:
                result = evaluate(scenario, family, test_seed, model=model,
                                  trace=seed == TRAINING_SEEDS[0], nurse_roster=nurse_roster)
                if weights is not None:
                    metrics = result['metrics']
                    difference = weights['waiting_minutes'] - REWARD_WEIGHTS['waiting_minutes']
                    metrics['raw_variant_reward'] = round(metrics['total_reward'] +
                                                          difference * metrics['waiting_minutes'], 4)
                episodes.append({'seed': test_seed, 'training_seed': seed, 'metrics': result['metrics']})
                if 'events' in result:
                    routes.append({**result, 'method': variant})
            print(f'{label} seed {seed}: {len(recorded)} training episodes, '
                  f'held-out mean {mean(r["metrics"]["total_reward"] for r in episodes[-len(TEST_SEEDS):]):.2f}',
                  flush=True)
            del model
        profiles = loss_profiles(losses, actual_steps)
        profiles.update(loss_profiles(terms, actual_steps))
        experiments[variant] = {'replicates': replications, 'episodes': episodes, 'profiles': profiles}
        if resume:
            temporary = partial.with_suffix('.tmp')
            temporary.write_text(json.dumps(family_result, separators=(',', ':'), allow_nan=False), encoding='utf-8')
            temporary.replace(partial)
    meta['ended_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    path = folder / f'{family}.json'
    path.write_text(json.dumps(family_result, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    if resume:
        partial.unlink(missing_ok=True)
    print(f'Saved {path}', flush=True)


def render_round(instances, output_dir, report, *, report_metadata=None):
    from .benchmark_report import render_report
    from .exploration_report import render_exploration

    payloads = []
    for instance in instances:
        tag = instance.lower()
        runs = [json.loads((output_dir / tag / f'{family}.json').read_text(encoding='utf-8'))
                for family in ALGORITHMS]
        first, second = [r['meta'] for r in runs]
        protocol = ('instance', 'steps', 'training_seeds', 'test_seeds')
        if any(first[key] != second[key] for key in protocol) or first['instance'] != instance:
            raise ValueError(f'Mismatched experiment protocols for {instance}')
        prior = json.loads((Path('output') / f'tuned-{tag}' / 'benchmark.json').read_text(encoding='utf-8'))
        if prior['meta']['test_seeds'] != first['test_seeds']:
            raise ValueError(f'Mismatched baseline test seeds for {instance}')
        baseline_ids = ('random', 'aco')
        baseline_methods = [m for m in prior['methods'] if m['id'] in baseline_ids]
        baseline_routes = [r for r in prior['runs'] if r['method'] in baseline_ids]
        methods = [{**m, 'family': 'baseline'} for m in baseline_methods]
        experiments = {m: {'episodes': [{'seed': r['seed'], 'training_seed': None, 'metrics': r['metrics']}
                                        for r in baseline_routes if r['method'] == m],
                           'replicates': [], 'profiles': {}} for m in baseline_ids}
        if report_metadata is not None:
            extra_meta = report_metadata(runs)
            fixed_wait_weight = extra_meta.get('baseline_waiting_weight', REWARD_WEIGHTS['waiting_minutes'])
            for route in baseline_routes:
                metrics = route['metrics']
                metrics['raw_variant_reward'] = round(metrics['total_reward'] + metrics['waiting_minutes'] *
                                                      (fixed_wait_weight - REWARD_WEIGHTS['waiting_minutes']), 4)
        for run in runs:
            methods.extend(run['methods'])
            experiments.update(run['experiments'])
        routes_href = f'{tag}-routes.html'
        intervals = {r['meta']['family']: {k: r['meta'][k] for k in ('started_at', 'ended_at')}
                     for r in runs}
        meta = {**{k: first[k] for k in protocol}, 'jobs': prior['meta']['jobs'], 'nurses': prior['meta']['nurses'],
                'reward_weights': REWARD_WEIGHTS, 'demo_seeds': DEMO_SEEDS, 'rollout_window': ROLLING_EPISODES,
                'devices': {r['meta']['family']: r['meta']['device'] for r in runs},
                'family_intervals': intervals,
                'family_times': {family: (datetime.fromisoformat(t['ended_at']) -
                                          datetime.fromisoformat(t['started_at'])).total_seconds()
                                 for family, t in intervals.items()},
                'validation_seeds': [], 'routes_href': routes_href,
                'note': ('Matched final training budgets and hyperparameters from the tuned study. '
                         'The fixed Random and ACO baselines reuse the original matching held-out traces. '
                         'Every learned variant is evaluated greedily on all three training seeds and 12 held-out seeds. '
                         'Route details show the predeclared training seed 11. The reward weights are fixed. '
                         'The short-wait teacher sorts legal visits by forecast waiting time, closing time, travel, and job ID. '
                         'Bounded ACO uses three ants, one iteration, and a lookahead of three visits.')}
        if report_metadata is not None:
            meta.update(extra_meta)
        payloads.append({'meta': meta, 'methods': methods, 'experiments': experiments})
        elapsed = (max(datetime.fromisoformat(r['meta']['ended_at']) for r in runs)
                   - min(datetime.fromisoformat(r['meta']['started_at']) for r in runs)).total_seconds()
        widths = ', '.join(f"{r['meta']['family'].upper()} {r['meta']['params'].get('width', 64)}"
                           for r in runs)
        route_meta = {**prior['meta'], **meta, 'best_rl': None, 'trials': 0, 'trial_steps': 0,
                      'final_steps': first['steps'], 'validation_seeds': [],
                      'parameters_from': 'the preceding tuned study',
                      'network_label': f'Units per hidden layer by agent. {widths}.',
                      'network_search': None, 'selection': meta['note'],
                      'runtime_seconds': elapsed,
                      'device': ', '.join(f'{family.upper()} on {device.upper()}'
                                          for family, device in meta['devices'].items())}
        route_methods = [{'id': m['id'], 'label': m['label'], 'description': m['description'],
                          'group': 'baseline' if m['family'] == 'baseline' else 'rl'} for m in methods]
        render_report({'meta': route_meta, 'scenario': prior['scenario'], 'methods': route_methods,
                       'tuning': {}, 'runs': baseline_routes + [route for run in runs for route in run['routes']]},
                      report.parent / routes_href)
    (output_dir / 'summary.json').write_text(json.dumps(payloads, separators=(',', ':'), allow_nan=False),
                                             encoding='utf-8')
    print(render_exploration(payloads, report))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance', choices=list('ABCD'), default='A')
    parser.add_argument('--family', choices=ALGORITHMS, default='dqn')
    parser.add_argument('--steps', type=int, default=12288)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output-dir', type=Path, default=Path('output/rounds/06-guided-exploration'))
    parser.add_argument('--render', nargs='+', choices=list('ABCD'))
    parser.add_argument('--report', type=Path, default=Path('docs/report/exploration/index.html'))
    args = parser.parse_args(argv)
    if args.render:
        render_round(args.render, args.output_dir, args.report)
        return
    if args.steps < 1:
        parser.error('Training steps must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('PyTorch cannot access CUDA')
    torch.set_num_threads(1)
    run_family(args.instance, args.family, args.steps, args.device, args.output_dir)


if __name__ == '__main__':
    main()
