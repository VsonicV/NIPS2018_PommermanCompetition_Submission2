"""Implementation of a simple deterministic agent using Docker."""

from pommerman import agents
from pommerman.runner import DockerAgentRunner
import logging
import json
from collections import namedtuple

import numpy as np

import sys
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




class MyAgent(DockerAgentRunner, policy):
    '''An example Docker agent class'''

    def __init__(self):
        self._agent = agents.SimpleAgent()

    def act(self, observation, action_space):
        return self._agent.act(observation, action_space)


def main():
    '''Inits and runs a Docker Agent'''
    agent = MyAgent()
    agent.run()


if __name__ == "__main__":
    main()
