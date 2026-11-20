"""Optional CPU multiprocess rollouts. This is not GPU SimPy simulation."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from .cli import run_episode


def _run_one(args):
    instance, policy, seed, limit = args
    _,m,_=run_episode(instance=instance,policy=policy,seed=seed,job_limit=limit)
    return m


def evaluate_parallel(*, instance='A',policy='greedy',seeds=(1,2,3,4),
                      job_limit=35, workers=2):
    jobs=[(instance,policy,seed,job_limit) for seed in seeds]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_one,jobs))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--instance',default='A',choices=list('ABCD'))
    ap.add_argument('--policy',default='greedy',choices=['greedy','random','aco'])
    ap.add_argument('--seeds',type=int,default=8)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--limit',type=int,default=35)
    a=ap.parse_args(argv)
    results=evaluate_parallel(instance=a.instance,policy=a.policy,
                     seeds=tuple(range(1,1+a.seeds)),job_limit=a.limit,workers=a.workers)
    for r in results:print(f"seed {r['seed']}: {r['jobs_completed']}/{r['jobs_total']} jobs, reward {r['total_reward']:.1f}")

if __name__=='__main__':main()
