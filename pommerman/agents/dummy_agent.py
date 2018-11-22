'''An agent that preforms a random action each step'''
from . import BaseAgent
import numpy as np


class DummyAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def act(self, obs, action_space):
        return 0 #action_space.sample()
