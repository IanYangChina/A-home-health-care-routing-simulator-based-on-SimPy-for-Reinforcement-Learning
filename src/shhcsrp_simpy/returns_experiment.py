"""Matched C/D experiments on waiting costs and longer RL return estimates."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .exploration import ALGORITHMS, REWARD_WEIGHTS, render_round, run_family


def configurations(instance, family, params):
    tuned = dict(params)
    if family == 'ppo':
        tuned.update(n_steps=512, gae_lambda=.95)
    controls = [
        ('control', 'control', .05, {}, 1, 'Original waiting cost and standard return estimator.'),
        ('wait15', 'wait 0.15', .15, {}, 1, 'Triples the waiting cost. Uses the control return estimator.'),
        ('wait30', 'wait 0.30', .30, {}, 1, 'Sextuples the waiting cost. Uses the control return estimator.'),
    ]
    longer = [
        ('n16', '16-step', .05, {}, 16, 'Accumulate up to 16 decision rewards before a masked value bootstrap.'),
        ('n32', '32-step', .05, {}, 32, 'Accumulate up to 32 decision rewards before a masked value bootstrap.'),
    ] if family == 'dqn' else [
        ('lambda1', 'λ=1', .05, {'gae_lambda': 1.}, 1,
         'Use λ=1 with the tuned discount. Rollouts ending mid-shift bootstrap at the buffer boundary.'),
        ('lambda1_gamma1', 'λ=1, γ=1', .05, {'gae_lambda': 1., 'gamma': 1.}, 1,
         'Use undiscounted reward within each rollout. Rollouts ending mid-shift still bootstrap.'),
    ]
    variants, settings = [], {}
    for suffix, label, cost, overrides, n_steps, description in controls + longer:
        variant = f'{family}_{suffix}'
        actual = {**tuned, **overrides}
        description += f" γ={actual['gamma']:.6f}."
        if family == 'ppo':
            description += f" λ={actual['gae_lambda']:g}. Rollout length {actual['n_steps']} decisions."
        variants.append((variant, f'{family.upper()} {label}', None, description))
        settings[variant] = {'params': actual, 'return_steps': n_steps,
                             'weights': {**REWARD_WEIGHTS, 'waiting_minutes': -cost}}
    return variants, settings


def report_metadata(runs):
    settings = {variant: setting for run in runs
                for variant, setting in run['meta']['variant_settings'].items()}
    return {
        'round': 'wait-and-returns',
        'variant_weights': {variant: setting['weights'] for variant, setting in settings.items()},
        'variant_settings': settings,
        'demo_seeds': [],
        'note': ('Each arm uses the same training seeds, decision budget, and tuned parameters within its agent and layout. '
                 'PPO uses 512-step rollouts in all arms. DQN uses uniform legal epsilon exploration. '
                 'All arms treat the modeled 480-minute shift end as terminal for learning. '
                 'Shortened DQN replay prefixes stop at the shift end. PPO with λ=1 still bootstraps at a mid-shift rollout boundary. '
                 'The stronger wait penalties and longer returns are tested separately. '
                 'The held-out score uses the original weights for every policy. '
                 'Route details show the predeclared training seed 11. Random and ACO use their matching saved test episodes.'),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance', choices='CD', default='C')
    parser.add_argument('--family', choices=ALGORITHMS, default='dqn')
    parser.add_argument('--steps', type=int, default=12288)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output-dir', type=Path, default=Path('output/rounds/07-wait-and-returns'))
    parser.add_argument('--render', nargs='+', choices='CD')
    parser.add_argument('--report', type=Path, default=Path('docs/report/returns/index.html'))
    args = parser.parse_args(argv)
    if args.render:
        render_round(args.render, args.output_dir, args.report, report_metadata=report_metadata)
        return
    if args.steps < 512 or args.steps % 512:
        parser.error('Use a positive multiple of 512 to match PPO and DQN training budgets')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('PyTorch cannot access CUDA')
    torch.set_num_threads(1)
    run_family(args.instance, args.family, args.steps, args.device, args.output_dir, configure=configurations)


if __name__ == '__main__':
    main()
