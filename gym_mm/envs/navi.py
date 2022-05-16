import gym
from gym.spaces import Box, Discrete
import numpy as np 
from copy import deepcopy

map_arr = [[0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,0,1,1,1,1,1,0],
    [0,1,1,1,1,1,0,1,1,1,1,1,0],
    [0,1,1,1,1,1,0,1,1,1,1,1,0],
    [0,1,1,1,1,1,0,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,0,1,1,1,1,1,0],
    [0,1,1,1,1,1,0,1,1,1,1,1,0],
    [0,1,1,1,1,1,0,1,1,1,1,1,0],
    [0,1,1,1,1,1,0,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0]]
v_block = 255/255
v_target = 191/255
v_agent = 63/255
map_arr = [[0,0,0,0,0],
    [0,1,0,1,0],
    [0,0,0,0,0],
    [0,1,0,1,0],
    [0,0,0,0,0]]

class Navi2D(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # Map settings
        self.map_h = 5
        self.map_w = 5

        # Parameters
        self.obs_shape = (self.map_h, self.map_w)

        self.observation_space = Box(low=0, high=255, shape=self.obs_shape, dtype=np.uint8)
        self.action_space = Discrete(4)

        self.max_steps = 200
        self.optimal_steps = 7

        self.terrain = np.array(map_arr)
        self.reset()

    def reset(self):
        self.steps = 0
        self.agent_pos = [0, 0]
        self.target_pos = [self.map_h-1, self.map_w-1]

        self.state = v_block * self.terrain
        self.state[tuple(self.agent_pos)] = v_agent
        self.state[tuple(self.target_pos)] = v_target

        return self.state
        
    def out_of_range(self, pos):
        if pos[0]<0 or pos[0]>=self.map_h or pos[1]<0 or pos[1]>=self.map_w:
            return True
        else:
            return False

    def step(self, action):
        new_pos = deepcopy(self.agent_pos)
        if action == 0:
            new_pos[0] += 1
        elif action == 1:
            new_pos[1] += 1
        elif action == 2:
            new_pos[0] -= 1
        else:
            new_pos[1] -= 1

        if self.out_of_range(new_pos):
            pass
        elif self.terrain[tuple(new_pos)] == 1:
            pass
        else:
            self.agent_pos = deepcopy(new_pos)

        self.state = v_block * self.terrain
        self.state[tuple(self.agent_pos)] = v_agent
        self.state[tuple(self.target_pos)] = v_target

        reward = 0
        info = {}
        if self.agent_pos == self.target_pos:
            done = True
            reward += 50
            if self.steps <= self.optimal_steps:
                reward += 50
        else:
            done = False
        reward -= 1
        self.steps += 1
        if self.steps >= self.max_steps:
            done = True
        
        return self.state, reward, done, info

    def get_agent_pos(self):
        pos = np.zeros(self.obs_shape)
        pos[tuple(self.agent_pos)] = 1.0
        return pos

    def render(self, mode='human', close=False):
        print('Current step: {0}/{1}'.format(self.steps, self.max_steps))