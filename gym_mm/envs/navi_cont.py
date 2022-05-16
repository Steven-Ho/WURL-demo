import gym
from gym.spaces import Box
import numpy as np 
from copy import deepcopy
import cv2

eps = 1e-6
class TreeNode:
    def __init__(self, idx, succ, p):
        self.idx = idx
        self.succ = succ
        self.p = p

class MinMaxTree:
    def __init__(self, inputs):
        self.root = TreeNode(None, [], "min")
        self._build_tree(self.root, inputs)

    def _build_tree(self, root, inputs):
        if root.p == "min":
            sp = "max"
        else:
            sp = "min"
        for x in inputs:
            if type(x) is int:
                c = TreeNode(x, [], sp)
                root.succ.append(c)
            else:
                c = TreeNode(None, [], sp)
                self._build_tree(c, x)
                root.succ.append(c)
    
    def evaluate(self, vals):
        return self._evaluate(self.root, vals)
    
    def _evaluate(self, root, vals):
        if root.succ == []:
            return vals[root.idx]
        else:
            if root.p == "min":
                r = True
                for x in root.succ:
                    r = r and self._evaluate(x, vals)
                    if not r:
                        break
            else:
                r = False
                for x in root.succ:
                    r = r or self._evaluate(x, vals)
                    if r:
                        break
            return r                    

class RuleSet:
    def __init__(self, A, b, rules):
        self.A = A
        self.b = b
        self.rules = rules
        self.num = A.shape[0]
        self.tree = MinMaxTree(rules)

    def in_region(self, point):
        c = np.matmul(self.A, point) - self.b
        v = np.transpose(c>=0.)[0].tolist()
        return self.tree.evaluate(v)
    
    def intersection(self, pos, vel):
        assert np.linalg.norm(vel)>0. 
        e = np.matmul(self.A, vel)
        f = self.b - np.matmul(self.A, pos)
        t = np.transpose(np.divide(f, e))[0]
        return t

    def first_contact(self, pos, vel):
        t = self.intersection(pos, vel)
        cd = []
        for x in t:
            if x<0 or x>1:
                cd.append(-1.)
            else:
                new_pos = pos + vel * (x + eps)
                if self.in_region(new_pos):
                    cd.append(-1.)
                else:
                    cd.append(x)
        idx = np.argsort(cd)
        fc_idx = []
        n = len(idx)
        min_d = -1.0
        for i in range(n):
            if cd[idx[i]]>=0:
                if fc_idx == []:
                    min_d = cd[idx[i]]
                    fc_idx.append(idx[i])
                elif cd[idx[i]] == min_d:
                    fc_idx.append(idx[i])
                else:
                    break
        return fc_idx, min_d

rules = [0,1,2,3,[4,5,6,7]]
A = np.array([[0., -1.], [0., 1.], [-1., 0.], [1., 0.], [0., 1.], [0., -1.], [1., 0.], [-1., 0.]])
b = np.array([[-1.], [0.], [-2.], [0.], [0.8], [-0.2], [1.8], [-0.2]])
init_pos = np.array([[0.1], [0.1]])
init_vel = np.array([[0.], [0.]])
target = np.array([[1.9], [0.9]])
rrules = [0,1,2,3]
rA = np.array([[0., 1.], [0., -1.], [1., 0.], [-1., 0.]])
rb = np.array([[0.8], [-1.], [1.8], [-2.]])

class Navi2DCont(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.A = A
        self.b = b
        self.rules = rules
        self.rs = RuleSet(A, b, rules)
        self.rA = rA
        self.rb = rb
        self.rrules = rrules
        self.rrs = RuleSet(rA, rb, rrules)
        self.dt = 0.1
        self.max_v = 0.02
        self.max_steps = 200

        self.obs_shape = (4,)
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
        self.acc = np.reshape(self.acc, (2,1))
        self.vel *= 0.9 # damping term
        self.vel += self.acc * self.dt
        self.vel = self._vclip(self.vel)
        p = deepcopy(self.pos)
        dp = deepcopy(self.vel) * self.dt
        reward = 0.
        while True:
            fc, d = self.rs.first_contact(p, dp)
            if fc == []:
                self.pos = p + dp
                self.trajectory.append(deepcopy(self.pos))
                self.vel = (dp/np.linalg.norm(dp))*np.linalg.norm(self.vel)
                break
            else:
                reward -= 1. # collision penalty
                n = np.array([[0.], [0.]])
                for idx in fc:
                    n += np.expand_dims(self.A[idx,:], axis=-1)
                n = n/np.linalg.norm(n)
                p += dp * d
                self.trajectory.append(deepcopy(p))
                dp = dp * (1-d)
                dp = dp - 1.01*np.dot(dp.T, n)*n
        
        state = self._state() 
        if self.rrs.in_region(self.pos):
            reward += 50.
            done = True
        else:
            reward -= np.linalg.norm(self.pos-target)*0.1
            done = False
        self.t += 1
        done = done or (self.t >= self.max_steps)
        info = {}
        return state, reward, done, info
    
    def _state(self):
        state = np.concatenate([self.pos, self.vel], axis=0)
        state = np.transpose(state)[0]
        return state

    def render(self, mode='human', close=False):
        img = np.zeros((512, 1024, 3), np.uint8)
        scale_f = 512.0
        n = len(self.trajectory)
        for i in range(n-1):
            s = self.trajectory[i]
            t = self.trajectory[i+1]
            s = (s * scale_f).T.astype(np.int16)[0].tolist()
            t = (t * scale_f).T.astype(np.int16)[0].tolist()
            cv2.line(img, tuple(s), tuple(t), color=(192, 156, 211), thickness=1)
        cv2.imwrite("example.jpg", img)