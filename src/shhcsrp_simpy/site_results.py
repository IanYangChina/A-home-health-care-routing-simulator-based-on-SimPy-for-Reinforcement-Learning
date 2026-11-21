"""Generate the project homepage from the latest measured experiment."""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
from statistics import mean


def render_latest(payloads: list[dict], output: Path) -> Path:
    meta = payloads[0]['meta']
    number, title, report, record, intro = {
        'guided-long-returns': (8, 'Guided exploration with longer returns', 'guided-returns', '08-guided-long-returns',
                                'DQN and PPO used guided exploration and a higher waiting cost on all four layouts.'),
        'full-roster-returns': (9, 'Full nurse roster and longer returns', 'roster-returns', '09-full-roster-returns',
                               'Compare two DQN return lengths and two PPO discounts with both guides on all four layouts.'),
        'attention-returns': (10, 'Matched return horizons and attention', 'attention-returns', '10-attention-returns',
                              'Compare 64-, 128-, and 256-decision horizons with and without attention on all four layouts.'),
    }[meta['round']]
    waiting_cost = f"{abs(meta['baseline_waiting_weight']):.2f}"
    panels = []
    for payload in payloads:
        meta = payload['meta']
        rows = []
        for method in payload['methods']:
            episodes = payload['experiments'][method['id']]['episodes']
            row = {'label': method['label'], 'family': method['family']}
            for key in ('jobs_completed', 'jobs_unserved', 'waiting_minutes', 'travel_minutes',
                        'total_reward', 'raw_variant_reward'):
                row[key] = mean(episode['metrics'][key] for episode in episodes)
            rows.append(row)
        panels.append({'instance': meta['instance'], 'jobs': meta['jobs'], 'nurses': meta['nurses'],
                       'routes': f"{report}/{meta['routes_href']}", 'rows': rows})

    comparison = []
    beats_aco, beats_random = [], []
    for panel in panels:
        rows = panel['rows']
        top = max((r for r in rows if r['family'] != 'baseline'), key=lambda r: r['raw_variant_reward'])
        random = next(r for r in rows if r['label'] == 'Random')
        aco = next(r for r in rows if r['family'] == 'baseline' and r['label'] != 'Random')
        if top['raw_variant_reward'] > aco['raw_variant_reward']:
            beats_aco.append(panel['instance'])
        if top['raw_variant_reward'] > random['raw_variant_reward']:
            beats_random.append(panel['instance'])
        cells = (top['label'], f"{top['jobs_completed']:.2f} / {panel['jobs']}",
                 f"{top['raw_variant_reward']:,.2f}", f"{random['raw_variant_reward']:,.2f}",
                 f"{aco['raw_variant_reward']:,.2f}")
        comparison.append(f'<tr><th scope="row">{escape(panel["instance"])}</th>' +
                          ''.join(f'<td>{escape(cell)}</td>' for cell in cells) + '</tr>')

    data = json.dumps(panels, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('<', r'\u003c')
    observations = ('The highest-scoring learned policy beat ACO on ' + ', '.join(beats_aco) + '.' if beats_aco else
                    'ACO scored at least as high as every learned policy on all four layouts.')
    observations += (' The highest-scoring learned policy beat Random on ' + ', '.join(beats_random) + '.' if beats_random else
                     ' Random scored at least as high as every learned policy on all four layouts.')
    output.parent.mkdir(parents=True, exist_ok=True)
    html = _HTML.replace('__TITLE__', escape(title)).replace('__ROUND__', str(number)).replace('__REPORT__', report)
    html = html.replace('__RECORD__', record).replace('__INTRO__', escape(intro)).replace('__OBSERVATIONS__', escape(observations))
    html = html.replace('__COST__', waiting_cost).replace('__SUMMARY__', ''.join(comparison)).replace('__DATA__', data)
    output.write_text(html, encoding='utf-8')
    return output


_HTML = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Latest results and a playable simulator for home care routing and dispatch.">
<title>Latest results · Home care dispatch</title>
<link rel="stylesheet" href="site.css">
</head>
<body>
<a class="skip-link" href="#main">Skip to content</a>
<header class="project-hero">
  <nav class="project-nav" aria-label="Project"><a href="index.html" aria-current="page">Latest results</a><a href="setup/index.html">Algorithm setup</a><a href="development/index.html">Development history</a><a href="simulator/index.html">Simulator</a></nav>
  <div class="project-container">
    <p class="project-kicker">Home care dispatch research</p>
    <h1>__TITLE__</h1>
    <p>__INTRO__</p>
    <div class="result-links"><a class="project-button secondary" href="simulator/index.html">Play the simulator</a><a class="project-button" href="__REPORT__/index.html#training">Explore training curves</a></div>
  </div>
</header>
<main class="project-container" id="main">
  <section class="project-section" aria-labelledby="overview-title">
    <h2 id="overview-title">Round __ROUND__ at a glance</h2>
    <p>__OBSERVATIONS__</p>
    <div class="project-scroll"><table class="project-table" aria-label="Best learned policy by layout at the __COST__ waiting cost">
      <thead><tr><th scope="col">Layout</th><th scope="col">Best learned policy</th><th scope="col">Visits completed</th><th scope="col">Learned reward</th><th scope="col">Random reward</th><th scope="col">ACO reward</th></tr></thead>
      <tbody>__SUMMARY__</tbody>
    </table></div>
  </section>
  <section class="project-section" aria-labelledby="compare-title">
    <h2 id="compare-title">Compare every policy</h2>
    <p class="project-muted">Learned means use three training seeds, each tested on 12 shifts. Fixed policies use the same 12 test seeds. Both reward columns apply a common waiting cost to all schedules. Travel and waiting are in nurse-minutes per shift.</p>
    <div class="result-controls"><label>Layout <select id="layout"></select></label><label>Chart metric <select id="metric"></select></label><a id="routes-link" href="__REPORT__/d-routes.html">Inspect selected routes and schedules</a></div>
    <p class="project-muted" id="layout-note" aria-live="polite"></p>
    <div id="policy-chart" class="project-card" aria-hidden="true"></div>
    <div class="project-scroll" style="margin-top:.75rem"><table class="project-table result-table" aria-label="Latest policy results">
      <thead><tr><th scope="col">Policy</th><th scope="col">Completed</th><th scope="col">Unserved</th><th scope="col">Travel min</th><th scope="col">Wait min</th><th scope="col">Reward at __COST__</th><th scope="col">Reward at 0.05</th></tr></thead>
      <tbody id="policy-results"></tbody>
    </table></div>
    <noscript><p>Enable JavaScript to switch layouts and view all policies.</p></noscript>
  </section>
  <section class="project-section" aria-label="Supporting evidence">
    <div class="project-grid">
      <article class="project-card"><h3><a href="__REPORT__/index.html">Training and reward profiles</a></h3><p>Inspect reward terms, losses, and variation across seeds.</p></article>
      <article class="project-card"><h3><a href="development/__RECORD__/index.html">Round __ROUND__ record</a></h3><p>Read the matched comparisons, computation and PI direction.</p></article>
      <article class="project-card"><h3><a href="benchmark/index.html">Tuned benchmark</a></h3><p>Compare the earlier tuned agents and four fixed policies.</p></article>
    </div>
  </section>
</main>
<footer class="project-footer"><div class="project-container">Stochastic home health care routing and scheduling</div></footer>
<script type="application/json" id="latest-data" data-waiting-cost="__COST__">__DATA__</script>
<script src="latest.js" defer></script>
</body>
</html>
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=Path('output/rounds/10-attention-returns/summary.json'))
    parser.add_argument('--output', type=Path, default=Path('docs/report/index.html'))
    args = parser.parse_args(argv)
    data = json.loads(args.summary.read_text(encoding='utf-8'))
    print(render_latest(data, args.output))


if __name__ == '__main__':
    main()
