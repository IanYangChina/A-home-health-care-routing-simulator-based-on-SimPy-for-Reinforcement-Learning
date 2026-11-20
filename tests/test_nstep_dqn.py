import numpy as np
import pytest

pytest.importorskip('stable_baselines3')

import gymnasium as gym
import torch

from shhcsrp_simpy.nstep_dqn import NStepDictReplayBuffer, NStepMaskedDQN, NStepSamples


def test_nstep_replay_shortens_at_episode_boundaries():
    observations = gym.spaces.Dict({'id': gym.spaces.Box(0., 50., shape=(1,))})
    buffer = NStepDictReplayBuffer(16, observations, gym.spaces.Discrete(2), device='cpu',
                                 n_steps=3, gamma=.5)

    def add(start, end, reward, *, done=False, timeout=False):
        current = {'id': np.array([[start]], dtype=np.float32)}
        following = {'id': np.array([[end]], dtype=np.float32)}
        buffer.add(current, following, np.array([1]), np.array([reward]), np.array([done]),
                   [{'TimeLimit.truncated': timeout}])

    add(0, 1, 1)
    add(1, 2, 2)
    assert buffer.size() == 0
    add(2, 3, 4)
    add(3, 4, 8, done=True)
    add(10, 11, 2)
    add(11, 12, 6, done=True, timeout=True)
    add(20, 21, 12)
    assert buffer.size() == 6

    samples = buffer._get_samples(np.arange(6))
    np.testing.assert_array_equal(samples.observations['id'].flatten(), [0, 1, 2, 3, 10, 11])
    np.testing.assert_array_equal(samples.next_observations['id'].flatten(), [3, 4, 4, 4, 12, 12])
    np.testing.assert_array_equal(samples.rewards.flatten(), [3, 6, 8, 8, 5, 6])
    np.testing.assert_array_equal(samples.discounts.flatten(), [.125, .125, .25, .5, .25, .5])
    np.testing.assert_array_equal(samples.dones.flatten(), [0, 1, 1, 1, 0, 0])


def test_nstep_target_bootstraps_with_the_correct_discount_and_mask():
    from stable_baselines3.common.logger import Logger

    class TinyDispatch(gym.Env):
        observation_space = gym.spaces.Dict({'eligible': gym.spaces.Box(0., 1., shape=(2,))})
        action_space = gym.spaces.Discrete(3)

    model = NStepMaskedDQN('MultiInputPolicy', TinyDispatch(), buffer_size=2,
                           policy_kwargs={'net_arch': []}, device='cpu')
    with torch.no_grad():
        model.q_net.q_net[-1].weight.zero_()
        model.q_net.q_net[-1].bias.zero_()
        model.q_net_target.q_net[-1].weight.zero_()
        model.q_net_target.q_net[-1].bias.copy_(torch.tensor([100., 8., 1.]))

    samples = NStepSamples(
        observations={'eligible': torch.tensor([[1., 0.], [1., 0.]])},
        actions=torch.tensor([[0], [0]]),
        next_observations={'eligible': torch.tensor([[0., 1.], [0., 0.]])},
        dones=torch.tensor([[0.], [1.]]),
        rewards=torch.tensor([[1.], [2.]]),
        discounts=torch.tensor([[.25], [.5]]),
    )
    model.replay_buffer.full = True
    model.replay_buffer.sample = lambda batch_size, env: samples
    model.set_logger(Logger(folder=None, output_formats=[]))
    model.train(gradient_steps=1, batch_size=2)

    assert model.logger.name_to_value['train/loss'] == pytest.approx(2.)
    model.get_env().close()
