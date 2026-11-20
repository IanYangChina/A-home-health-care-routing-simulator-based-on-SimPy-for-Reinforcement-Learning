"""Optuna search space and Stable-Baselines3 settings for the benchmark."""


WIDTHS = (64, 128, 256)
BATCH_SIZES = (32, 64, 128)
TARGET_UPDATE_INTERVALS = (256, 512, 1024, 2048)
ROLLOUT_LENGTHS = (128, 256, 512)


def suggest_params(trial, method, *, tune_width=True):
    params = {'width': trial.suggest_categorical('width', WIDTHS)} if tune_width else {}
    params.update({
        'batch_size': trial.suggest_categorical('batch_size', BATCH_SIZES),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 2e-3, log=True),
        'gamma': trial.suggest_float('gamma', .95, .999),
    })
    if method == 'dqn':
        params['target_update_interval'] = trial.suggest_categorical(
            'target_update_interval', TARGET_UPDATE_INTERVALS)
        params['exploration_fraction'] = trial.suggest_float('exploration_fraction', .15, .6)
    elif method == 'ppo':
        params['n_steps'] = trial.suggest_categorical('n_steps', ROLLOUT_LENGTHS)
        params['ent_coef'] = trial.suggest_float('ent_coef', .001, .035, log=True)
    else:
        raise ValueError(f'Unknown benchmark method {method!r}')
    return params


def training_settings(method, params, steps, *, network=(64, 64)):
    """Translate a sampled trial or an older fixed-parameter record to SB3 kwargs."""
    params = dict(params)
    width = params.pop('width', None)
    architecture = [width, width] if width is not None else list(network)
    settings = {'policy_kwargs': {'net_arch': architecture}, 'batch_size': 64}
    if method == 'dqn':
        settings.update(buffer_size=min(steps, 25_000), learning_starts=min(256, steps // 4),
                        train_freq=4, gradient_steps=1, target_update_interval=512,
                        exploration_final_eps=.05)
    elif method == 'ppo':
        settings.update(n_epochs=5, n_steps=128)
    else:
        raise ValueError(f'Unknown benchmark method {method!r}')
    settings.update(params)
    return settings
