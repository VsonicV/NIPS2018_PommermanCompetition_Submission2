'''An agent that preforms a random action each step'''
from . import BaseAgent
import numpy as np


class RandomAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def act(self, obs, action_space):
        self._character.is_random = True
        return np.random.randint(5) #action_space.sample()
