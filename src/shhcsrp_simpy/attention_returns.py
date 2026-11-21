"""Round 10. Match nominal return horizons and test dispatch attention."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .exploration import ALGORITHMS, DEMO_SEEDS, REWARD_WEIGHTS, VARIANTS, render_round, run_family
from .returns_experiment import report_metadata as return_metadata


HORIZONS = (64, 128, 256)
PPO_GAMMA = .9995
PPO_ROLLOUT = 512


def gae_lambda(horizon):
    """Set the undisrupted geometric weight mass, 1 / (1 - gamma * lambda)."""
    return (1. - 1. / horizon) / PPO_GAMMA


def configurations(instance, family, params):
    variants, settings = [], {}
    for horizon in HORIZONS:
        if family == 'dqn':
            tuned = dict(params)
            return_steps = horizon
            title = f'{horizon}-step'
        else:
            tuned = {**params, 'gamma': PPO_GAMMA, 'gae_lambda': gae_lambda(horizon), 'n_steps': PPO_ROLLOUT}
            return_steps = 1
            title = f'GAE {horizon}'
        for key, label, strategy, description in VARIANTS[family][1:]:
            for architecture in ('mlp', 'attention'):
                attention = architecture == 'attention'
                variant = f'{key}_n{horizon}_{architecture}'
                variants.append((variant, f'{label} · {title} · {"attention" if attention else "MLP"}', strategy,
                                 description + (' Appends current-nurse attention over pending jobs and the full roster.'
                                                if attention else ' Uses the original flattened observation.')))
                settings[variant] = {'params': tuned, 'return_steps': return_steps, 'nurse_roster': True,
                                     'attention': attention, 'weights': {**REWARD_WEIGHTS, 'waiting_minutes': -.30}}
    return variants, settings


def report_metadata(runs):
    return {
        **return_metadata(runs),
        'round': 'attention-returns',
        'baseline_waiting_weight': -.30,
        'demo_seeds': DEMO_SEEDS,
        'reused_dqn_64_mlp': runs[0]['meta'].get('reused_dir') is not None,
        'observation_normalization': ('Both architectures use the same scaled appointment, clock, eligibility and '
                                      'full-roster inputs. Attention adds pooled job and nurse features.'),
        'note': ('DQN compares 64, 128, and 256 recorded decision rewards at its tuned discount. PPO uses '
                 '512-decision rollouts and gamma 0.9995. Its lambda sets the geometric weight mass of GAE '
                 'to 64, 128, or 256 before rollout and shift boundaries. Attention adds separate job and nurse '
                 'summaries to the MLP inputs. Each variant uses short-wait or ACO guidance and charges 0.30 points '
                 'per appointment waiting minute. Test shifts use the same seeds and greedy dispatch. Both reported '
                 'waiting costs apply to all policies. Example routes use training seed 11.'),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance', choices='ABCD', default='A')
    parser.add_argument('--family', choices=ALGORITHMS, default='dqn')
    parser.add_argument('--steps', type=int, default=12288)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--reuse-round9', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('output/rounds/10-attention-returns'))
    parser.add_argument('--render', nargs='+', choices='ABCD')
    parser.add_argument('--report', type=Path, default=Path('docs/report/attention-returns/index.html'))
    args = parser.parse_args(argv)
    if args.render:
        render_round(args.render, args.output_dir, args.report, report_metadata=report_metadata)
        return
    if args.steps < PPO_ROLLOUT or args.steps % PPO_ROLLOUT:
        parser.error('Use a positive multiple of 512 to match PPO and DQN training budgets')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('PyTorch cannot access CUDA')
    torch.set_num_threads(1)
    run_family(args.instance, args.family, args.steps, args.device, args.output_dir,
               configure=configurations, reuse_dir=args.reuse_round9, resume=True)


if __name__ == '__main__':
    main()
