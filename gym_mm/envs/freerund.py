import gym
from gym.spaces import Box, Discrete
import numpy as np 
from copy import deepcopy
import cv2

class FreeRunD(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.pos = np.array([0, 0])
        self.max_steps = 100
        self.boundaries = 100

        self.obs_shape = (2,)
        self.observation_space = Box(low=-self.boundaries, high=self.boundaries, shape=self.obs_shape, dtype=np.uint8)
        self.action_space = Discrete(5)
        self.reset()

    def reset(self):
        self.pos = np.array([0, 0])
        self.t = 0
        self.trajectory = []
        return self._state()

    def _state(self):
        return deepcopy(self.pos)

    def _sclip(self):
        outflow_u = self.pos > self.boundaries
        outflow_d = self.pos < -self.boundaries
        self.pos = self.pos*(1-outflow_u) + self.boundaries*outflow_u
        self.pos = self.pos*(1-outflow_d) - self.boundaries*outflow_d

    def step(self, action):
        self.action = deepcopy(action)
        if self.action == 1:
            self.pos[0] -= 1
        elif self.action == 2:
            self.pos[0] += 1
        elif self.action == 3:
            self.pos[1] -= 1
        elif self.action == 4:
            self.pos[1] += 1
        self._sclip()
        state = self._state()
        reward = 0
        info = {}
        self.t += 1
        done = (self.t == self.max_steps)
        if done:
            reward = abs(self.pos[0]) + abs(self.pos[1])
        self.trajectory.append(deepcopy(self.pos))
        return state, reward, done, info

    def render(self, mode='human', close=False, message=None): 
        img = np.zeros((1024, 1024, 3), np.uint8)
        block_width = 5
        n = len(self.trajectory)
        center = np.array([512, 512])
        for i in range(n-1):
            s = self.trajectory[i]
            t = self.trajectory[i+1]
            s = (block_width * s + center).astype(np.int16).tolist()
            t = (block_width * t + center).astype(np.int16).tolist()
            cv2.line(img, tuple(s), tuple(t), color=(192, 156, 211), thickness=3)
        if message is not None:
            cv2.putText(img, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (104, 242, 18), 2)
        cv2.imwrite("example.jpg", img)