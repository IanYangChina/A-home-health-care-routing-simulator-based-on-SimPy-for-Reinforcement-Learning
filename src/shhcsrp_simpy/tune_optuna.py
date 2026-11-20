"""Optional Optuna search over established SB3 MaskablePPO hyperparameters.

Test seeds (51-53) differ from training seeds; final benchmarks should use a
separate holdout seed set, not the Optuna validation seeds.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .train_sb3 import make_model, evaluate


def main(argv=None):
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna missing. Run pip install -e '.[rl]'") from exc
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--instance',choices=list('ABCD'),default='A')
    ap.add_argument('--limit',type=int,default=35)
    ap.add_argument('--trials',type=int,default=8)
    ap.add_argument('--steps',type=int,default=4000)
    ap.add_argument('--output',default='output/optuna_results.json')
    args=ap.parse_args(argv)

    def objective(trial):
        n_steps=trial.suggest_categorical('n_steps',[64,128,256])
        batch_size=trial.suggest_categorical('batch_size',[32,64])
        params={'learning_rate':trial.suggest_float('learning_rate',1e-5,3e-3,log=True),
                'gamma':trial.suggest_categorical('gamma',[.95,.98,.99,.995]),
                'ent_coef':trial.suggest_float('ent_coef',1e-5,.05,log=True)}
        model,scenario=make_model(instance=args.instance,job_limit=args.limit,
                   seed=100+trial.number,n_steps=n_steps,batch_size=batch_size,**params)
        # Evaluate after each training block, so Optuna pruning has a real
        # intermediate score rather than being a no-op pruner configuration.
        interval=max(500,args.steps//3)
        trained=0
        score=0.0
        while trained<args.steps:
            chunk=min(interval,args.steps-trained)
            model.learn(total_timesteps=chunk,reset_num_timesteps=False)
            trained+=chunk
            score,results=evaluate(model,scenario,seeds=(51,52,53))
            trial.report(score,step=trained)
            if trial.should_prune():
                model.get_env().close()
                raise optuna.TrialPruned()
        trial.set_user_attr('per_seed_rewards',[r['total_reward'] for r in results])
        model.get_env().close()
        return score

    study=optuna.create_study(direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3))
    study.optimize(objective,n_trials=args.trials)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'best_value':study.best_value,
                               'best_params':study.best_params,
                               'trials':[{'number':t.number,'value':t.value,
                                          'params':t.params} for t in study.trials]},indent=2))
    print('Best validation reward:',study.best_value)
    print('Best params:',study.best_params)
    print('Saved:',path)

if __name__=='__main__':main()
