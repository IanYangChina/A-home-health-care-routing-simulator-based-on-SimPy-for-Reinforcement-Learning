"""Round 9. Full nurse roster and longer returns with guided exploration."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .exploration import ALGORITHMS, DEMO_SEEDS, REWARD_WEIGHTS, VARIANTS, render_round, run_family
from .returns_experiment import report_metadata as return_metadata


def configurations(instance, family, params):
    variants, settings = [], {}
    horizons = (32, 64) if family == 'dqn' else (.998, .9995)
    for horizon in horizons:
        if family == 'dqn':
            suffix, title = f'n{horizon}', f'{horizon}-step'
            tuned, return_steps = dict(params), horizon
        else:
            suffix, title = f'g{str(horizon).split(".")[1]}', f'γ={horizon:g}'
            tuned, return_steps = {**params, 'gamma': horizon, 'n_steps': 512, 'gae_lambda': 1.}, 1
        for key, label, strategy, description in VARIANTS[family][1:]:
            variant = f'{key}_{suffix}'
            variants.append((variant, f'{label} · {title}', strategy, description))
            settings[variant] = {'params': tuned, 'return_steps': return_steps, 'nurse_roster': True,
                                 'weights': {**REWARD_WEIGHTS, 'waiting_minutes': -.30}}
    return variants, settings


def report_metadata(runs):
    return {
        **return_metadata(runs),
        'round': 'full-roster-returns',
        'baseline_waiting_weight': -.30,
        'demo_seeds': DEMO_SEEDS,
        'observation_normalization': ('Clock, job and nurse features use fixed scaling. The nurse roster includes '
                                      'each nurse’s phase, assignment and forecast time to availability.'),
        'note': ('All learned arms observe the full nurse roster and charge 0.30 points per appointment waiting minute. '
                 'Release times use the service-duration forecast. DQN compares 32- and 64-decision returns at the tuned discount. '
                 'PPO compares discounts 0.998 and 0.9995 with λ=1 and 512-decision rollouts. '
                 'The shift end is terminal, and a PPO rollout ending mid-shift bootstraps. '
                 'DQN mixes short-wait or bounded ACO guidance into epsilon exploration. PPO uses those teachers for imitation '
                 'before on-policy training. Final learned policies use greedy dispatch on the same 12 test seeds as Random and ACO. '
                 'Both reported waiting costs apply to every policy. Example routes use the predeclared training seed 11.'),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance', choices='ABCD', default='A')
    parser.add_argument('--family', choices=ALGORITHMS, default='dqn')
    parser.add_argument('--steps', type=int, default=12288)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output-dir', type=Path, default=Path('output/rounds/09-full-roster-returns'))
    parser.add_argument('--render', nargs='+', choices='ABCD')
    parser.add_argument('--report', type=Path, default=Path('docs/report/roster-returns/index.html'))
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
