"""Simple multi-seed baseline comparison with outcomes shown per seed."""
from statistics import mean, stdev
from shhcsrp_simpy.cli import run_episode

for policy in ('random','greedy','aco'):
    outcomes=[]
    for seed in (11, 12, 13, 14, 15):
        _,metrics,_=run_episode(instance='A',job_limit=35,policy=policy,seed=seed)
        outcomes.append(metrics)
    scores=[m['jobs_completed'] for m in outcomes]
    print(f'{policy:7s} completed {mean(scores):.1f} ± {stdev(scores):.1f} '
          f'jobs of {outcomes[0]["jobs_total"]}')
