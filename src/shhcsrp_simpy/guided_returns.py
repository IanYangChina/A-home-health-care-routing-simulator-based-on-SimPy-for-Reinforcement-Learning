"""Combine guided exploration, longer returns, and a higher waiting cost on A–D."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .exploration import ALGORITHMS, DEMO_SEEDS, REWARD_WEIGHTS, VARIANTS, render_round, run_family
from .returns_experiment import report_metadata as return_metadata


def configurations(instance, family, params):
    tuned = dict(params)
    if family == 'ppo':
        tuned.update(n_steps=512, gae_lambda=1.)
    steps = (32 if instance == 'C' else 16) if family == 'dqn' else 1
    weights = {**REWARD_WEIGHTS, 'waiting_minutes': -.15}
    return VARIANTS[family], {variant: {'params': tuned, 'return_steps': steps, 'weights': weights}
                              for variant, *_ in VARIANTS[family]}


def report_metadata(runs):
    return {
        **return_metadata(runs),
        'round': 'guided-long-returns',
        'baseline_waiting_weight': -.15,
        'demo_seeds': DEMO_SEEDS,
        'note': ('All learned arms charge 0.15 points per nurse-minute of appointment waiting. '
                 'DQN uses 32-step returns on C and 16-step returns on A, B, and D. '
                 'PPO uses λ=1 and 512-step rollouts, with a value bootstrap at mid-shift boundaries. '
                 'Both agents retain their tuned discounts, and the shift end is terminal for learning. '
                 'Exploration controls and both guides use the same settings within each family and layout. '
                 'DQN mixes guidance into epsilon exploration. PPO imitates demonstrations and then collects its own rollouts. '
                 'Held-out scores use both waiting costs for every policy. Random and ACO are rescored from matching saved test episodes. '
                 'Route details show the predeclared training seed 11.'),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance', choices=list('ABCD'), default='A')
    parser.add_argument('--family', choices=ALGORITHMS, default='dqn')
    parser.add_argument('--steps', type=int, default=12288)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output-dir', type=Path, default=Path('output/rounds/08-guided-long-returns'))
    parser.add_argument('--render', nargs='+', choices=list('ABCD'))
    parser.add_argument('--report', type=Path, default=Path('docs/report/guided-returns/index.html'))
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
