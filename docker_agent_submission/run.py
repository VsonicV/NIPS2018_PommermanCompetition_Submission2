"""Implementation of a simple deterministic agent using Docker."""

from pommerman import agents
from pommerman.runner import DockerAgentRunner
import logging
import json
from collections import namedtuple

import numpy as np

import sys
import os
sys.path.append('../tools/')

import policies
import tf_util
import random

import pommerman
from pommerman import agents

logger = logging.getLogger(__name__)

Config = namedtuple('Config', [
    'l2coeff', 'noise_stdev', 'episodes_per_batch', 'timesteps_per_batch',
    'calc_obstat_prob', 'eval_prob', 'snapshot_freq',
    'return_proc_mode', 'episode_cutoff_mode'
])

def make_session(single_threaded):
    import tensorflow as tf
    if not single_threaded:
        return tf.InteractiveSession()
    return tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1))


class MyAgent(DockerAgentRunner):
    '''An example Docker agent class'''

    def __init__(self, policy):
        self._agent = agents.ESAgent(policy=policy)

    def act(self, observation, action_space):
        return self._agent.act(observation, action_space)


def main():
    '''Inits and runs a Docker Agent'''
    #exp_file = "../configurations/pommerman.json"
    exp_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pommerman.json')
    mydir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'snapshot_iter01660_agent.h5')
    with open(exp_file, 'r') as f:
        exp = json.loads(f.read())

    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
    ]
    env = pommerman.make('PommeTeamCompetitionFast-v0', agent_list)
    sess = make_session(single_threaded=True)

    loaded_policy = getattr(policies, exp['policy']['type'])(mydir, env.observation_space, env.action_space, **exp['policy']['args'])
    policy = getattr(policies, exp['policy']['type'])(None, env.observation_space, env.action_space, **exp['policy']['args'])
    tf_util.initialize()

    loaded_policy.initialize_from(mydir)

    theta = loaded_policy.get_trainable_flat()
    policy.set_trainable_flat(theta)

    agent = MyAgent(policy)
    agent.run()

    env.close()

if __name__ == "__main__":
    main()
