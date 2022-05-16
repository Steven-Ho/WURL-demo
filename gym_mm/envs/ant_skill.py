import gym
from gym.spaces import Box, Discrete
import numpy as np 
from copy import deepcopy
import cv2
from .freerun import FreeRun
from algo.sac import SACTrainer
from algo.utils import wrapped_obs
from functools import reduce
from .ant_custom_env import AntCustomEnv

class AntSkill(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, args):
        self.lower_env = AntCustomEnv()
        self.lower_env.seed(args.seed)
        obs_shape_list = self.lower_env.observation_space.shape
        obs_shape = reduce((lambda x,y: x*y), obs_shape_list)
        action_space = self.lower_env.action_space

        prefix = args.prefix
        self.args = args
        if args.backend in ["diayn", "dads"]:
            self.trainers = [SACTrainer(obs_shape, action_space, args) for _ in range(args.num_modes)]
            for x in range(len(self.trainers)):
                self.trainers[x].load_model(prefix+"sac_actor_{}_{}_{}".format(args.scenario, args.backend, x), prefix+"sac_critic_{}_{}_{}".format(args.scenario, args.backend, x))     
        else:
            self.trainers = [SACTrainer(obs_shape, action_space, args) for _ in range(args.num_modes)]
            for x in range(len(self.trainers)):
                self.trainers[x].load_model(prefix+"sac_actor_{}_wurl_{}_{}".format(args.scenario, args.backend, x), prefix+"sac_critic_{}_wurl_{}_{}".format(args.scenario, args.backend, x))

        self.obs_shape = (obs_shape,)
        self.action_n = args.num_modes
        self.observation_space = Box(low=-10.0, high=10.0, shape=self.obs_shape)
        self.action_space = Discrete(self.action_n)
        self.max_steps = 50
        self.reset()

    def reset(self):
        self.t = 0
        self.state = self.lower_env.reset()
        # self.goal = np.array(self.lower_env.goal())
        # self.state = np.concatenate([self.state, self.goal], axis=0)
        return self.state      

    def step(self, action):
        self.skill = deepcopy(action)

        obs = self.lower_env._state()
        init_pos = obs[:2]
        accumulated_reward = 0
        all_done = False
        for _ in range(self.args.max_episode_len):
            centered_obs = deepcopy(obs)
            centered_obs[:2] -= init_pos
            # if self.args.backend in ["diayn", "dads"]:
            #     if type(self.skill) is not np.ndarray:
            #         self.skill = np.array([self.skill])
            #     wobs = wrapped_obs(centered_obs, self.skill, self.args.num_modes)
            #     action, _ = self.trainers.act(wobs, eval=True)
            #     if len(action.shape) > 1:
            #         action = action[0]
            # else:
            action, _ = self.trainers[self.skill].act(centered_obs, eval=True)
            if len(action.shape) > 1:
                action = action[0]
                # if discrete_action:
                #     action = action[0]  
            new_obs, reward, done, info = self.lower_env.step(action)
            # self.lower_env.render()
            accumulated_reward += reward        
            obs = new_obs
            if done:
                all_done = True
                break  

        self.state = deepcopy(new_obs)
        # self.goal = np.array(self.lower_env.goal())
        # self.state = np.concatenate([self.state, self.goal], axis=0)
        
        info = {}
        self.t += 1
        all_done = all_done or (self.t == self.max_steps)

        return self.state, accumulated_reward, all_done, info

    def render(self, mode='human', close=False, message=None):
        self.lower_env.render()
