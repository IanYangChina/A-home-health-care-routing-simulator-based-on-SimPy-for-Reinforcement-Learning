"""Optional Stable-Baselines3 example. Install pip install -e '.[rl]'.

MaskablePPO is used because vanilla SB3 PPO does NOT enforce Gym action
masks. We implement the environment, NOT a new RL algorithm.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from .data import load_scenario
from .gym_env import SHHCSRPGymEnv
from .viz import export_html


def make_model(*, instance='A', job_limit=35, seed=11, n_steps=128,
               batch_size=64, learning_rate=3e-4, gamma=.99, ent_coef=.01):
    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise RuntimeError("RL dependencies missing. Run: pip install -e '.[rl]'") from exc
    scenario=load_scenario(instance,job_limit=job_limit)
    def make_env():
        return SHHCSRPGymEnv(scenario=scenario)
    vec=DummyVecEnv([make_env])
    model=MaskablePPO('MultiInputPolicy',vec,verbose=0,seed=seed,
                      n_steps=n_steps,batch_size=batch_size,learning_rate=learning_rate,
                      gamma=gamma,ent_coef=ent_coef)
    return model,scenario


def evaluate(model, scenario, *, seeds=(51,52,53), save_report=None):
    records=[]
    for seed in seeds:
        env=SHHCSRPGymEnv(scenario=scenario)
        obs,info=env.reset(seed=seed)
        reward_total=0.
        ended=False
        while not ended:
            action,_=model.predict(obs,deterministic=True,action_masks=info['action_mask'])
            obs,reward,terminated,truncated,info=env.step(int(action))
            ended=terminated or truncated
            reward_total+=reward
        m=env.model.metrics()
        m.update({'seed':seed,'total_reward':round(reward_total,3),'policy':'MaskablePPO'})
        records.append(m)
        if save_report is not None and seed==seeds[0]:
            export_html(scenario,env.model.events,m,save_report,
                        policy='MaskablePPO',total_reward=reward_total)
        env.close()
    score=float(np.mean([r['total_reward'] for r in records]))
    return score, records


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--instance',choices=list('ABCD'),default='A')
    ap.add_argument('--limit',type=int,default=35)
    ap.add_argument('--steps',type=int,default=5000)
    ap.add_argument('--seed',type=int,default=11)
    ap.add_argument('--output-dir',default='output/sb3')
    args=ap.parse_args(argv)
    model,scenario=make_model(instance=args.instance,job_limit=args.limit,seed=args.seed)
    model.learn(total_timesteps=args.steps)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    model.save(out/'maskable_ppo')
    score,records=evaluate(model,scenario,seeds=(51,52,53),save_report=out/'evaluation.html')
    (out/'evaluation.json').write_text(json.dumps({'mean_reward':score,'runs':records},indent=2))
    print('Held-out evaluation reward:',score)
    print('Saved model and HTML:',out)

if __name__=='__main__':main()
