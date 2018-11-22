'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents
import numpy as np

'''
New feature: 9*9 map with following values:
Passage = 0
Rigid/out of boundary = 1
Wood = 2
Enemy = 3
ExtraBomb = 4
IncrRange = 5
Kick = 6
Teammate = 7
Bomb = 8 + (9-bomb_life)*3  + (blast_range-2)
Flames = 35
plus three scalar characteristics:
player ammo counts
blast strength
can_kick
'''
def feature_engineering(obs):
    board_size_original = 11
    view_range = 4
    bomb_life_max = 9
    blast_range_min = 2
    blast_range_max = view_range
    board_in_view = np.ones((2 * view_range + 1, 2 * view_range + 1))
    for row in range(2 * view_range + 1):
        for col in range(2 * view_range + 1):
            row_original = obs['position'][0] - view_range + row
            col_original = obs['position'][1] - view_range + col
            if row_original < 0 or row_original > board_size_original -1 or col_original < 0 or col_original > board_size_original -1:
                value_cur = 1
            else:
                value_cur = obs['board'][row_original][col_original]
                assert value_cur != 5
                # bomb
                if value_cur == 3 or obs['bomb_blast_strength'][row_original][col_original] > 0:
                    value_cur = 8 + (bomb_life_max - obs['bomb_life'][row_original][col_original]) * (blast_range_max - blast_range_min + 1) \
                    + (obs['bomb_blast_strength'][row_original][col_original] - blast_range_min)
                elif value_cur == 4:
                # flames
                    value_cur = 35
                elif value_cur >= 6 and value_cur <=8:
                # power-ups
                    value_cur -= 2
                elif value_cur > 8:
                # agents
                    enemies_list = [enemy_agent.value for enemy_agent in obs['enemies']]
                    if value_cur in enemies_list:
                        value_cur = 3
                    else:
                        value_cur = 7
            board_in_view[row][col] = value_cur
    obs_modified = np.ravel(board_in_view)
    info_additional = []
    if obs['can_kick']:
        info_additional.append(1)
    else:
        info_additional.append(0)
    info_additional.append(obs['blast_strength'])
    info_additional.append(obs['ammo'])
    obs_modified = np.append(obs_modified, info_additional)
    return obs_modified

def main():
    '''Simple function to bootstrap a game.

       Use this as an example to set up your training env.
    '''
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    # Create a set of agents (exactly four)
    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        #agents.PlayerAgent(agent_control="arrows"),
        #agents.PlayerAgent(agent_control="wasd"),
        #agents.RandomAgent(),
        #agents.SimpleAgent(),
        #agents.RandomAgent(),
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)
    #env = pommerman.make('PommeTeamCompetition-v0', agent_list)

    # Run the episodes just like OpenAI Gym
    for i_episode in range(1):
        state = env.reset()
        done = False
        i = 0
        while not done:
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            print('time_step {}'.format(i))
            print('actions: {}'.format(actions))
            print('state: {}'.format(state))
            print('features_engineered: {}, num_features: {}'.format(feature_engineering(state[0]), len(feature_engineering(state[0]))))
        print('Episode {} finished'.format(i_episode))
    env.close()


if __name__ == '__main__':
    main()
