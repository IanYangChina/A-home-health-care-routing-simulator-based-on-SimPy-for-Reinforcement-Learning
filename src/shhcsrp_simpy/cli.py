"""CLI demonstration, shared evaluation protocol, and report generation."""
from __future__ import annotations
import argparse
import json
import csv
from pathlib import Path
import numpy as np

from .data import load_scenario
from .gym_env import SHHCSRPGymEnv
from .policies import RandomPolicy, GreedyPolicy, ACOPolicy
from .viz import export_html


def run_episode(*, instance='A', policy='greedy', seed=7, data_dir=None,
                job_limit=None, max_decisions=5000, duration_cv=.15,
                verbose=False, **policy_kw):
    scenario=load_scenario(instance,root=data_dir,job_limit=job_limit)
    env=SHHCSRPGymEnv(scenario=scenario,duration_cv=duration_cv)
    models={'greedy':GreedyPolicy,'random':RandomPolicy,'aco':ACOPolicy}
    if policy not in models:
        raise ValueError('Supported policies: '+', '.join(models))
    agent=models[policy](seed=seed,**policy_kw) if policy!='greedy' else GreedyPolicy()
    obs,info=env.reset(seed=seed)
    total_reward=0.
    decisions=0
    terminated=truncated=False
    while not (terminated or truncated):
        if decisions>=max_decisions:
            raise RuntimeError('Reached safety limit on decisions; check scheduler progress')
        action=agent.act(obs,info)
        obs,reward,terminated,truncated,info=env.step(action)
        total_reward+=reward
        decisions+=1
        if verbose and decisions%20==0:
            print(f'  decision {decisions}: {env.render()}')
    assert env.model is not None
    metrics=env.model.metrics()
    metrics.update({'policy':policy,'seed':seed,'decisions':decisions,
                    'total_reward':round(total_reward,4)})
    return scenario,metrics,list(env.model.events)


def save_result(scenario, metrics, events, outbase: str | Path):
    base=Path(outbase)
    base.parent.mkdir(parents=True,exist_ok=True)
    metrics_path=base.with_suffix('.json')
    events_path=base.with_name(base.name+'_events.csv')
    report_path=base.with_suffix('.html')
    metrics_path.write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    keys=['time','type','nurse','job','detail']
    with events_path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys,extrasaction='ignore')
        writer.writeheader();writer.writerows(events)
    export_html(scenario,events,metrics,report_path,policy=metrics['policy'],
                total_reward=metrics['total_reward'])
    return metrics_path,events_path,report_path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance',choices=list('ABCD'),default='A')
    parser.add_argument('--policy',choices=['random','greedy','aco'],default='aco')
    parser.add_argument('--compare',action='store_true',help='run all three baselines with identical seed')
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--limit',type=int,default=35,help='first N appointment rows; 0=full instance')
    parser.add_argument('--data-dir',type=Path,default=None)
    parser.add_argument('--output',default='output/shhcsrp_demo')
    parser.add_argument('--duration-cv',type=float,default=.15)
    parser.add_argument('--verbose',action='store_true')
    args=parser.parse_args(argv)
    policies=['random','greedy','aco'] if args.compare else [args.policy]
    results=[]
    for p in policies:
        scenario,metrics,events=run_episode(instance=args.instance,policy=p,
            seed=args.seed,job_limit=(None if args.limit==0 else args.limit),
            data_dir=args.data_dir,duration_cv=args.duration_cv,verbose=args.verbose)
        outbase=args.output+'_'+p if args.compare else args.output
        files=save_result(scenario,metrics,events,outbase)
        results.append(metrics)
        print(json.dumps(metrics,indent=2))
        print('Saved:', ' | '.join(str(f) for f in files))
    if args.compare:
        path=Path(args.output+'_comparison.csv')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
        print('Comparison:',path)

if __name__=='__main__':
    main()
