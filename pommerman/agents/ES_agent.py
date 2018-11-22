'''An agent that preforms a random action each step'''
from __future__ import division
from . import BaseAgent
from .. import constants
from .. import characters
import numpy as np

import logging
logger = logging.getLogger(__name__)

class ESAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def __init__(self, character=characters.Bomber, policy=None):
        super(ESAgent, self).__init__(character)
        self._policy = policy

    '''
    New feature before one-hot coding: 9*9 map with following values:
    Passage = 0
    Rigid/Out of boundary = 1
    Wood = 2
    Bomb_life = (bomb_life_max + 1 - bomb_life)/bomb_life_max # if there is no bomb, the life in original obs will be 0, but we set them to bomb_life_max+1, so that the input value will be 0
    Flames = 4
    Bomb_blast_strength = blast_strength/blast_range_max   #(blast_strength + 1 - blast_range_min)/(blast_range_max-blast_range_min+1)
    Powerups = 0.5*ExtraBomb + 0.667*IncrRange + 1.0*Kick
    Teammate = 7
    Enemy = 8
    '''
    def _feature_engineering(self, obs):
        ob_max_value = 13
        board_size_original = 11
        view_range = 4
        bomb_life_max = 9
        blast_range_min = 2
        blast_range_max = view_range
        board_in_view = np.ones((2 * view_range + 1, 2 * view_range + 1))
        strength_in_view = np.zeros((2 * view_range + 1, 2 * view_range + 1))
        life_in_view = np.zeros((2 * view_range + 1, 2 * view_range + 1))
        enemies_list = obs['enemies']
        for row in range(2 * view_range + 1):
            for col in range(2 * view_range + 1):
                row_original = obs['position'][0] - view_range + row
                col_original = obs['position'][1] - view_range + col
                if row_original < 0 or row_original > board_size_original -1 or col_original < 0 or col_original > board_size_original -1:
                    value_cur = 1
                else:
                    value_cur = obs['board'][row_original][col_original]
                    assert value_cur != 5
                    if value_cur > 8:
                        if value_cur in enemies_list:
                            value_cur = 10
                        else:
                            value_cur = 9
                    strength_in_view[row][col] = obs['bomb_blast_strength'][row_original][col_original]
                    life_in_view[row][col] = obs['bomb_life'][row_original][col_original]
                    if life_in_view[row][col] == 0:
                        life_in_view[row][col] = bomb_life_max + 1
                board_in_view[row][col] = value_cur
        ob_hot = np.eye(11)[board_in_view.astype(np.uint8)]
        ob_hot[:, :, 6] = ob_hot[:, :, 6] * 0.5 + ob_hot[:, :, 7] * 0.667 + ob_hot[:, :, 8]
        ob_hot = np.delete(ob_hot, [7, 8], axis=2)
        ob_hot[:, :, 3] = (bomb_life_max + 1 - life_in_view.astype(np.float32)) / bomb_life_max
        ob_hot[:, :, 5] = strength_in_view.astype(np.float32) / blast_range_max  #(obs["bomb_blast_strength"].astype(np.float32) + 1 - blast_range_min) / (blast_range_max - blast_range_min + 1)

        #ob_hot = ob_hot.transpose((2, 0, 1))
        obs_modified = np.ravel(ob_hot)
        info_additional = []
        if obs['can_kick']:
            info_additional.append(1.0)
        else:
            info_additional.append(0.0)
        info_additional.append((obs['blast_strength'] - blast_range_min)/(blast_range_max - blast_range_min))
        info_additional.append(obs['ammo']/10.0)
        obs_modified = np.append(obs_modified, info_additional)

        return obs_modified


    def act(self, obs, action_space):
        epsilon = 0.2
        obs_modified = self._feature_engineering(obs)
        ac_scores = self._policy.act(obs_modified[None])[0][0]

        for i in range(len(ac_scores)):
            if ac_scores[i] < 0.15: #!= np.amax(ac_scores):
                ac_scores[i] = 0
        #logger.info("ac_scores: {}".format(ac_scores))

        #print("ac_scores: {}".format(ac_scores))
        ac_probs = ac_scores/ac_scores.sum()
        #logger.info("ac_probs: {}".format(ac_probs))
        if np.random.rand() >= epsilon:
            ac = np.argmax(ac_scores)
        else:
            ac = np.random.choice(len(ac_scores), 1, p=ac_probs)[0] #np.argmax(ac_scores)  #np.random.choice(len(ac_scores), 1, p=ac_probs)[0]
        #logger.info("action: {}".format(ac))
        #print("agent{}, action: {}, prob: {}".format(self._character.agent_id, ac, ac_probs[ac]))
        return int(ac)
