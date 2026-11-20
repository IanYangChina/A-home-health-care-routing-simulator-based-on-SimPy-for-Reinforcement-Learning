"""Acceptance and regression checks (requires installation of package)."""
import json
import numpy as np
import pytest

from shhcsrp_simpy.data import DEPOT, Job, Scenario, load_scenario, travel_minutes
from shhcsrp_simpy.gym_env import SHHCSRPGymEnv
from shhcsrp_simpy.policies import GreedyPolicy, ACOPolicy, EarliestDeadlinePolicy
from shhcsrp_simpy.cli import run_episode
from shhcsrp_simpy.viz import export_html


def test_original_dataset_columns_and_subsetting():
    scenario=load_scenario('A',job_limit=12)
    assert len(scenario.jobs)==12
    assert len(scenario.nurse_skills)==7
    assert len({j.id for j in scenario.jobs})==12
    assert min(j.elder_id for j in scenario.jobs)>=1
    assert all(j.window_start<j.window_end for j in scenario.jobs)
    assert np.all(scenario.preferences>0)


def test_original_full_instance_lengths():
    assert [len(load_scenario(x).jobs) for x in 'ABCD']==[60,60,159,303]


def test_travel_and_skill_compatibility():
    s=load_scenario('A',job_limit=12)
    assert travel_minutes(s.depot,s.depot,s.speed)==0.
    env=SHHCSRPGymEnv(scenario=s)
    obs,info=env.reset(seed=1)
    assert env.observation_space.contains(obs)
    assert env.action_space.n==13
    # First nurse has skill 1, so only level-1 jobs can be dispatched.
    mask=info['action_mask']
    assert all(not mask[j.id] for j in s.jobs if j.demand>1)
    env.close()


def test_episode_invariants_and_reproducibility():
    s1,m1,e1=run_episode(instance='A',job_limit=18,policy='greedy',seed=13)
    s2,m2,e2=run_episode(instance='A',job_limit=18,policy='greedy',seed=13)
    assert m1==m2
    assert e1==e2
    assert m1['jobs_completed']+m1['jobs_unserved']==18
    served=[e['job'] for e in e1 if e['type']=='service_end']
    assert len(served)==len(set(served))
    assert len(served)==m1['jobs_completed']
    assert [e['time'] for e in e1]==sorted(e['time'] for e in e1)
    for event in e1:
        if event['type']=='service_start':
            assert s1.nurse_skills[event['nurse']]>=s1.jobs[event['job']].demand
            assert event['time']<=s1.jobs[event['job']].window_end+1e-5


def test_aco_produces_valid_dispatched_jobs():
    env=SHHCSRPGymEnv(instance='A',job_limit=14)
    obs,info=env.reset(seed=9)
    agent=ACOPolicy(ants=4,iterations=2,seed=9)
    done=False
    while not done:
        act=agent.act(obs,info)
        assert info['action_mask'][act]
        obs,rew,terminated,truncated,info=env.step(act)
        done=terminated or truncated
    assert env.model.metrics()['jobs_completed']>=0
    env.close()


def test_earliest_deadline_and_common_random_numbers():
    scenario = Scenario('paired', (
        Job(0, 1, 1, DEPOT, 0., 200.),
        Job(1, 2, 1, DEPOT, 0., 100.),
    ), (1,), np.ones((2, 1)))
    env = SHHCSRPGymEnv(scenario=scenario, common_random_numbers=True)
    samples = []
    for order in ((0, 1), (1, 0)):
        obs, info = env.reset(seed=17)
        assert EarliestDeadlinePolicy().act(obs, info) == 1
        for action in order:
            assert info['action_mask'][action]
            _, _, terminated, truncated, info = env.step(action)
        assert terminated and not truncated
        samples.append({e['job']: e['duration'] for e in env.model.events
                        if e['type'] == 'service_start'})
    assert samples[0] == samples[1]
    assert samples[0][0] != samples[0][1]
    env.close()


def test_wait_penalty_and_shift_end_change_learning_signals_only():
    scenario = Scenario('fixed shift', (Job(0, 1, 1, DEPOT, 8., 30.),),
                        (1,), np.ones((1, 1)), horizon=33.)
    outputs = []
    for penalty, terminal in ((.05, False), (.30, True)):
        env = SHHCSRPGymEnv(scenario=scenario, common_random_numbers=True,
                           wait_penalty=penalty, terminal_shift=terminal)
        _, info = env.reset(seed=1)
        assert info['action_mask'][0]
        _, reward, terminated, truncated, info = env.step(0)
        outputs.append((reward, terminated, truncated, info, env.model.events))
        env.close()
    baseline, modified = outputs
    assert baseline[1:3] == (False, True)
    assert modified[1:3] == (True, False)
    assert baseline[3]['metrics'] == modified[3]['metrics']
    assert baseline[4] == modified[4]
    assert baseline[3]['metrics']['jobs_unserved'] == 1
    assert modified[0] - baseline[0] == pytest.approx(-2.)


def test_html_has_charts_and_trace(tmp_path):
    s,m,events=run_episode(instance='A',job_limit=12,policy='greedy',seed=3)
    path=export_html(s,events,m,tmp_path/'report.html',policy='greedy')
    txt=path.read_text(encoding='utf-8')
    assert 'Plotly.newPlot' in txt
    assert 'Event trace' in txt
    assert 'Nurse activities' in txt


def test_full_instance_aco_and_large_greedy_invariants():
    for instance,policy in [('A','aco'),('D','greedy')]:
        s,m,e=run_episode(instance=instance,job_limit=None,policy=policy,seed=17)
        served=[x['job'] for x in e if x['type']=='service_end']
        assert len(served)==len(set(served))
        assert len(served)+m['jobs_unserved']==m['jobs_total']
        assert m['time']<=s.horizon+1e-5
