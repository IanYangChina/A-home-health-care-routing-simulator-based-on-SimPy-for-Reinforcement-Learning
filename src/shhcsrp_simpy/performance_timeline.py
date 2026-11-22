"""Build the development chart from the committed evaluation reports."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
from statistics import mean, stdev


METHODS = {
    'random': 'Random', 'greedy': 'Greedy', 'edf': 'Earliest deadline',
    'aco': 'Ant colony', 'dqn': 'DQN', 'ppo': 'PPO',
}
METRICS = {
    'jobs_completed': 'Served visits', 'jobs_unserved': 'Unserved visits',
    'travel_minutes': 'Travel (nurse-minutes)', 'waiting_minutes': 'Waiting (nurse-minutes)',
    'completion_percent': 'Completion (%)', 'service_minutes': 'Service (nurse-minutes)',
    'time': 'Shift elapsed (minutes)', 'decisions': 'Dispatch decisions',
    'reward_030': 'Reward at 0.30 wait cost', 'total_reward': 'Reward at 0.05 wait cost',
}
EXPLORATION_REPORTS = {
    6: 'exploration', 7: 'returns', 8: 'guided-returns',
    9: 'roster-returns', 10: 'attention-returns',
}


def embedded_data(path, script_id):
    pattern = rf'<script type="application/json" id="{script_id}">(.*?)</script>'
    match = re.search(pattern, path.read_text(encoding='utf-8'), re.DOTALL)
    if match is None:
        raise ValueError(f'Missing {script_id} in {path}')
    return json.loads(match[1])


def summarize(episodes):
    samples = {key: [] for key in METRICS}
    for episode in episodes:
        m = episode['metrics']
        values = {**m, 'reward_030': m['total_reward'] - .25 * m['waiting_minutes'],
                  'completion_percent': 100 * m['jobs_completed'] / m['jobs_total']}
        for key, rows in samples.items():
            rows.append(values[key])
    return {key: {'mean': mean(rows), 'sd': stdev(rows) if len(rows) > 1 else None}
            for key, rows in samples.items()}


def timeline_data(development):
    rounds, instances = [], {layout: {} for layout in 'ABCD'}
    for path in sorted(development.glob('*/manifest.json')):
        manifest = json.loads(path.read_text(encoding='utf-8'))
        number = manifest['round']
        families = manifest.get('family_runs') or manifest.get('guided_family_runs') or []
        measured = manifest.get('measured', [])
        interval = manifest.get('compute_interval')
        if interval:
            starts, ends = [interval['started_at']], [interval['ended_at']]
        else:
            starts = [r['started_at'] for r in families] or [r['first_optuna_log'] for r in measured
                                                            if r.get('first_optuna_log')]
            ends = [r['ended_at'] for r in families] or [r['results_file_time'] for r in measured]
        runner_seconds = [(datetime.fromisoformat(r['ended_at']) - datetime.fromisoformat(r['started_at'])).total_seconds()
                          for r in families] or [r['runtime_seconds'] for r in measured]
        info = {'number': number, 'title': manifest['title'], 'start': min(starts, default=None),
                'end': max(ends, default=None), 'layouts': [], 'note': manifest['timing'],
                'estimated': interval['estimated'] if interval else bool(measured and not families),
                'compute_seconds': sum(runner_seconds) if runner_seconds else None}
        rounds.append(info)
        candidates = {}
        if number in (3, 4):
            for artifact in measured:
                report = embedded_data(path.parent / artifact['public_report'], 'benchmark-data')
                layout = report['meta']['instance']
                for method in report['methods']:
                    key = method['id']
                    label = method['label']
                    if key in ('dqn', 'ppo'):
                        label += (f" · {artifact['label'].split(' · ')[-1]}" if number == 3
                                  else ' · tuned checkpoint')
                    episodes = [ep for ep in report['runs'] if ep['method'] == key]
                    candidates.setdefault((layout, key), []).append((label, episodes))
        elif number in EXPLORATION_REPORTS:
            report = development.parent / EXPLORATION_REPORTS[number] / 'index.html'
            for experiment in embedded_data(report, 'exploration-data'):
                layout = experiment['meta']['instance']
                for method in experiment['methods']:
                    family = method['family'] if method['family'] != 'baseline' else method['id']
                    episodes = experiment['experiments'][method['id']]['episodes']
                    candidates.setdefault((layout, family), []).append((method['label'], episodes))
        for (layout, method), variants in candidates.items():
            if method not in METHODS:
                raise ValueError(f'Unknown timeline method {method!r} in Round {number}')
            options = [(summarize(episodes), label, len(episodes)) for label, episodes in variants if episodes]
            if not options:
                continue
            best = max(options, key=lambda row: row[0]['reward_030']['mean']) if method in ('dqn', 'ppo') else options[0]
            stats, label, count = best
            instances[layout].setdefault(method, []).append({'round': number, 'variant': label,
                                                            'n': count, 'metrics': stats})
        info['layouts'] = [layout for layout in instances if any(key[0] == layout for key in candidates)]
    return {'methods': METHODS, 'metrics': METRICS, 'rounds': rounds, 'instances': instances}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development', type=Path, default=Path('docs/report/development'))
    args = parser.parse_args(argv)
    data = timeline_data(args.development)
    target = args.development / 'performance-data.js'
    target.write_text('window.developmentPerformance = ' + json.dumps(data, ensure_ascii=False, allow_nan=False,
                                                                     separators=(',', ':')) + ';\n', encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
