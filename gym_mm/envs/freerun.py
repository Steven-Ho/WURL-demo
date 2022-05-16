import gym
from gym.spaces import Box
import numpy as np 
from copy import deepcopy
import cv2

GOALS = [(-5.0, 5.0), (-5.0,-5.0), (5.0,-5.0)]
GOAL_RADIUS = 1.0
class FreeRun(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, max_steps=100):
        self.pos = np.array([0.0, 0.0])
        self.vel = np.array([0.0, 0.0])
        self.dt = 0.1
        self.max_v = 1.0
        self.max_steps = max_steps
        self.boundaries = 10. # symmetric boundries for X and Y
        self.curr_goal = 0

        self.obs_shape = (4,)
        self.action_shape = (2,)
        self.observation_space = Box(low=-10.0, high=10.0, shape=self.obs_shape)
        self.action_space = Box(low=-1.0, high=1.0, shape=self.action_shape)
        self.reset()

    def reset(self):
        self.pos = np.array([0.0, 0.0])
        self.vel = np.array([0.0, 0.0])
        self.t = 0
        self.trajectory = []
        self.curr_goal = 0
        return self._state()
    
    def reset_vel(self):
        self.vel = np.array([0.0, 0.0])

    def _state(self):
        return np.concatenate([self.pos, self.vel], axis=0)

    def _vclip(self, v):
        if np.linalg.norm(v) > self.max_v:
            return v/np.linalg.norm(v)
        else:
            return v        
    def goal(self):
        if self.curr_goal < len(GOALS):
            return GOALS[self.curr_goal]
        else:
            return GOALS[0]

    # def step(self, action):
    #     self.acc = deepcopy(action)
    #     self.acc = np.reshape(self.acc, (2,))
    #     self.vel *= 0.98 # damping term
    #     self.vel += self.acc * self.dt
    #     self.vel = self._vclip(self.vel)
    #     self.pos += self.vel * self.dt
    #     state = self._state()
    #     reward = -1./20 # Time penalty
        
    #     info = {}
    #     self.t += 1
    #     done = (self.t == self.max_steps)
    #     if self.curr_goal < len(GOALS):
    #         goal_dist = np.linalg.norm(np.array(GOALS[self.curr_goal]) - self.pos)
    #         if goal_dist < GOAL_RADIUS:
    #             reward += 50
    #             self.curr_goal += 1
    #     else:
    #         done = True
    #     if self.curr_goal < len(GOALS):
    #         info = {"goal": GOALS[self.curr_goal]}
    #     else:
    #         info = {"goal": GOALS[0]}
    #     # if done:
    #     #     reward = np.linalg.norm(self.pos)
    #     self.trajectory.append(deepcopy(self.pos))
    #     return state, reward, done, info
    def step(self, action):
        self.acc = deepcopy(action)
        self.acc = np.reshape(self.acc, (2,))
        self.vel *= 0.98 # damping term
        self.vel += self.acc * self.dt
        self.vel = self._vclip(self.vel)
        self.pos += self.vel * self.dt
        state = self._state()
        reward = 0
        info = {}
        self.t += 1
        done = (self.t == self.max_steps)
        if done:
            reward = np.linalg.norm(self.pos)
        self.trajectory.append(deepcopy(self.pos))
        return state, reward, done, info
        
    def render(self, mode='human', close=False, message=None):
        img = np.zeros((1024, 1024, 3), np.uint8)
        scale_f = 51.2
        n = len(self.trajectory)
        for i in range(n-1):
            s = self.trajectory[i]
            t = self.trajectory[i+1]
            s = ((s + self.boundaries) * scale_f).astype(np.int16).tolist()
            t = ((t + self.boundaries) * scale_f).astype(np.int16).tolist()
            cv2.line(img, tuple(s), tuple(t), color=(192, 156, 211), thickness=1)
        if message is not None:
            cv2.putText(img, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (104, 242, 18), 2)
        cv2.imwrite("example.jpg", img)