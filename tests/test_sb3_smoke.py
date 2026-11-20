"""Optional SB3 compatibility and stable action masking tests."""
import pytest

pytest.importorskip('sb3_contrib')
pytest.importorskip('optuna')

import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env

from shhcsrp_simpy.benchmark import StableDispatch, loss_profiles
from shhcsrp_simpy.exploration import RewardTerms, rolling_terms
from shhcsrp_simpy.guided_returns import configurations
from shhcsrp_simpy.gym_env import SHHCSRPGymEnv
from shhcsrp_simpy.masked_dqn import MaskedDQN


def test_sb3_gym_interface():
    env = SHHCSRPGymEnv(instance='A', job_limit=12)
    check_env(env, warn=True, skip_render_check=True)
    env.close()


def test_stable_dispatch_keeps_the_appointment_id():
    env = StableDispatch(SHHCSRPGymEnv(instance='A', job_limit=12))
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    defer = env.action_space.n - 1
    for _ in range(20):
        np.testing.assert_array_equal(obs['eligible'], info['action_mask'][:-1])
        np.testing.assert_array_equal(env.action_masks(), info['action_mask'])
        if obs['eligible'].any():
            break
        obs, _, terminated, truncated, info = env.step(defer)
        assert not (terminated or truncated or info['invalid_action'])
    job_id = int(np.flatnonzero(obs['eligible'])[-1])
    events = env.unwrapped.model.events
    index = len(events)
    _, _, _, _, info = env.step(job_id)
    assert not info['invalid_action']
    assert events[index]['type'] == 'dispatch' and events[index]['job'] == job_id
    env.close()


def test_dqn_masks_online_target_and_random_actions(tmp_path):
    env = StableDispatch(SHHCSRPGymEnv(instance='A', job_limit=12))
    model = MaskedDQN('MultiInputPolicy', env, device='cpu', seed=1, buffer_size=16)
    obs, _ = env.reset(seed=1)
    batch = {key: np.stack([value, value, value]) for key, value in obs.items()}
    batch['eligible'][:] = 0
    batch['eligible'][0, 2] = 1
    batch['eligible'][1, 7] = 1
    defer = env.action_space.n - 1
    legal = np.column_stack((batch['eligible'].astype(bool), np.ones(3, dtype=bool)))
    tensor_obs, _ = model.policy.obs_to_tensor(batch)
    for network in (model.q_net, model.q_net_target):
        with torch.no_grad():
            last_layer = network.q_net[-1]
            last_layer.weight.zero_()
            last_layer.bias.copy_(torch.arange(env.action_space.n))
            last_layer.bias[-1] = -1
            values = network(tensor_obs)
        assert np.isneginf(values.numpy()[~legal]).all()
        np.testing.assert_array_equal(values.argmax(dim=1).numpy(), [2, 7, defer])

    predicted, _ = model.predict(batch, deterministic=True)
    np.testing.assert_array_equal(predicted, [2, 7, defer])
    model._last_obs = batch
    model.exploration_rate = 1.
    for _ in range(10):
        sampled, _ = model.predict(batch)
        actions, replay_actions = model._sample_action(learning_starts=1, n_envs=3)
        assert legal[np.arange(3), sampled].all()
        assert legal[np.arange(3), actions].all()
        np.testing.assert_array_equal(actions, replay_actions)

    model.save(tmp_path / 'masked')
    loaded = MaskedDQN.load(tmp_path / 'masked.zip', device='cpu')
    np.testing.assert_array_equal(loaded.predict(batch, deterministic=True)[0], predicted)
    alone = {key: value[0] for key, value in batch.items()}
    assert loaded.predict(alone, deterministic=True)[0].item() == 2
    env.close()


def test_loss_profiles_weight_each_training_seed_equally():
    histories = [
        {128: {'train/loss': 1.}, 256: {'train/loss': 3.}, 512: {'train/loss': 5.},
         600: {'train/loss': 8.}},
        {512: {'train/loss': 9.}},
    ]
    first, last = loss_profiles(histories, 768)['train/loss']
    assert first == {'steps': 512, 'mean': 6., 'std': pytest.approx(4.242640687), 'n': 2}
    assert last == {'steps': 768, 'mean': 8., 'std': None, 'n': 1}


def test_training_reward_profiles_preserve_signed_contributions():
    counter = RewardTerms()
    counter.num_timesteps = 600
    counter.locals = {'dones': [True], 'infos': [{
        'metrics': {'jobs_completed': 3, 'jobs_unserved': 1,
                    'travel_minutes': 10., 'waiting_minutes': 20.},
        'episode': {'r': 20.7},
    }]}
    assert counter._on_step()
    rows = rolling_terms(counter.episodes, 768)
    assert list(rows) == [768]
    assert rows[768]['completed_weighted'] == 30.
    assert rows[768]['unserved_weighted'] == -8.
    assert rows[768]['travel_weighted'] == pytest.approx(-.3)
    assert rows[768]['waiting_weighted'] == -1.
    assert rows[768]['reward'] == pytest.approx(rows[768]['realized_reward'])


@pytest.mark.parametrize('instance,family,return_steps', [('A', 'dqn', 16), ('C', 'dqn', 32),
                                                          ('D', 'ppo', 1)])
def test_combined_round_matches_settings_across_exploration_arms(instance, family, return_steps):
    original = {'gamma': .98, 'n_steps': 128} if family == 'ppo' else {'gamma': .98}
    variants, settings = configurations(instance, family, original)
    assert [strategy for _, _, strategy, _ in variants] == [None, 'short_wait', 'aco']
    control = settings[variants[0][0]]
    assert all(arm == control for arm in settings.values())
    assert control['return_steps'] == return_steps
    assert control['weights']['waiting_minutes'] == -.15
    assert control['params']['gamma'] == original['gamma']
    if family == 'ppo':
        assert original['n_steps'] == 128
        assert (control['params']['n_steps'], control['params']['gae_lambda']) == (512, 1.)
