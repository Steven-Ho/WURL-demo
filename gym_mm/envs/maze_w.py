import gym
from gym.spaces import Box
import numpy as np 
from copy import deepcopy
import cv2
from .maze_utils import BlockMap

eps = 1e-6
# vertex in counter-clockwise order -> top-left, bottom-left, bottom-right, top-right
blocks = [
    [[0.,-1.], [0.,0.], [2.,0.], [2.,-1.]],
    [[-1.,0.], [-1.,2.], [0.,2.], [0.,0.]],
    [[0.,2.], [0.,3.], [2.,3.], [2.,2.]],
    [[2.,0.], [2.,2.], [3.,2.], [3.,0.]], # 4 out boundaries
    [[0.4,0.], [0.4,0.8], [0.8,0.8], [0.8,0.]],
    [[1.2,0.], [1.2,1.2], [1.6,1.2], [1.6,0.]],
    [[0.4,1.2], [0.4,1.6], [1.6,1.6], [1.6,1.2]]
]

init_pos = np.array([0.2, 0.4])
init_vel = np.array([0., 0.])

class MazeW(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.map = BlockMap(blocks)

        self.dt = 0.33
        self.max_v = 0.5
        self.max_steps = 100

        self.obs_shape = (4,)
        # self.obs_shape = (2,)
        self.action_shape = (2,)
        self.observation_space = Box(low=0.0, high=2.0, shape=self.obs_shape)
        self.action_space = Box(low=-1.0, high=1.0, shape=self.action_shape)
        self.reset()

    def reset(self):
        self.pos = init_pos
        self.vel = init_vel
        self.acc = np.array([[0.], [0.]])
        self.trajectory = [init_pos]
        self.t = 0
        state = self._state() 
        return state

    def _vclip(self, v):
        if np.linalg.norm(v) > self.max_v:
            return (v/np.linalg.norm(v))*self.max_v
        else:
            return v

    def step(self, action):
        self.acc = deepcopy(action)
        self.acc = np.reshape(self.acc, (2,))
        self.vel *= 0.0 # damping term
        self.vel += self.acc * self.dt
        self.vel = self._vclip(self.vel)
        p = deepcopy(self.pos)
        dp = deepcopy(self.vel) * self.dt
        reward = 0.
        tr, dp = self.map.reflected_point(p, dp)
        self.trajectory += tr
        self.vel = (dp/(np.linalg.norm(dp)+eps))*np.linalg.norm(self.vel)
        self.pos = tr[-1] + dp
        self.t += 1
        state = self._state()
        done = (self.t >= self.max_steps)
        info = {}
        return state, reward, done, info
    
    def _state(self):
        state = np.concatenate([self.pos, self.vel], axis=0)
        # state = deepcopy(self.pos)
        return state

    def render(self, mode='human', close=False, message=None):
        img = np.zeros((1024, 1024, 3), np.uint8)
        img = self.map.render(img)
        scale_f = 512.0
        boundaries = 0.0
        n = len(self.trajectory)
        for i in range(n-1):
            s = self.trajectory[i]
            t = self.trajectory[i+1]
            s = ((s + boundaries) * scale_f).astype(np.int16).tolist()
            t = ((t + boundaries) * scale_f).astype(np.int16).tolist()
            cv2.line(img, tuple(s), tuple(t), color=(192, 156, 211), thickness=2)
        if message is not None:
            cv2.putText(img, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (104, 242, 18), 2)
        cv2.imwrite("maze.jpg", img)