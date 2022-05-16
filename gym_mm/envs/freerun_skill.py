import gym
from gym.spaces import Box, Discrete
import numpy as np 
from copy import deepcopy
import cv2
from .freerun import FreeRun
from algo.sac import SACTrainer
from algo.utils import wrapped_obs
from functools import reduce

class Oracle():
    def __init__(self) -> None:
        self.actions = [[1.0, 1.0], [1.0, 0.0], [1.0, -1.0], [0.0, -1.0], [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
        return

    def act(self, skill):
        return np.array(self.actions[skill])

class FreeRunSkill(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, args):
        self.lower_env = FreeRun(max_steps=1000)
        self.lower_env.seed(None)
        obs_shape_list = self.lower_env.observation_space.shape
        obs_shape = reduce((lambda x,y: x*y), obs_shape_list)
        action_space = self.lower_env.action_space

        prefix = args.prefix
        self.args = args
        if args.backend in ["diayn", "dads", "diayn_indie", "dads_indie"]:
            if args.backend in ["diayn_indie", "dads_indie"]:
                self.trainers = [SACTrainer(obs_shape, action_space, args) for _ in range(args.num_modes)]
                for x in range(len(self.trainers)):
                    self.trainers[x].load_model(prefix+"sac_actor_{}_{}_{}".format(args.scenario, args.backend, x), prefix+"sac_critic_{}_{}_{}".format(args.scenario, args.backend, x))     
            else:
                self.trainers = SACTrainer(obs_shape + args.num_modes, action_space, args)
                self.trainers.load_model(prefix+"sac_actor_{}_{}".format(args.scenario, args.backend), prefix+"sac_critic_{}_{}".format(args.scenario, args.backend))
        elif args.backend == "oracle":
            self.trainers = Oracle()
        else:
            self.trainers = [SACTrainer(obs_shape, action_space, args) for _ in range(args.num_modes)]
            for x in range(len(self.trainers)):
                self.trainers[x].load_model(prefix+"sac_actor_{}_wasserstein_{}_{}".format(args.scenario, args.backend, x), prefix+"sac_critic_{}_wasserstein_{}_{}".format(args.scenario, args.backend, x))

        self.obs_shape = (4+2,)
        self.action_n = args.num_modes
        self.observation_space = Box(low=-10.0, high=10.0, shape=self.obs_shape)
        self.action_space = Discrete(self.action_n)
        self.max_steps = 50
        # self.goals = [np.array([-5.0, 5.0]), np.array([-5.0, -5.0]), np.array([5.0, -5.0])]
        # self.goals = [np.array([-5.0, 5.0])]
        # self.curr_goal = 0
        # self.tol = 1.0 # tolerance from goals
        self.reset()

    def reset(self):
        self.t = 0
        # self.curr_goal = 0
        self.state = self.lower_env.reset()
        self.goal = np.array(self.lower_env.goal())
        self.state = np.concatenate([self.state, self.goal], axis=0)
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
            if self.args.backend in ["diayn", "dads"]:
                if type(self.skill) is not np.ndarray:
                    self.skill = np.array([self.skill])
                wobs = wrapped_obs(centered_obs, self.skill, self.args.num_modes)
                action, _ = self.trainers.act(wobs, eval=False)
                if len(action.shape) > 1:
                    action = action[0]
            elif self.args.backend == "oracle":
                action = self.trainers.act(self.skill)
            else:
                action, _ = self.trainers[self.skill].act(centered_obs, eval=False)
                if len(action.shape) > 1:
                    action = action[0]
                # if discrete_action:
                #     action = action[0]  
            new_obs, reward, done, info = self.lower_env.step(action)
            accumulated_reward += reward        
            obs = new_obs
            if done:
                all_done = True
                break  
        # self.lower_env.reset_vel()
        # new_obs = self.lower_env._state()
        # curr_pos = new_obs[:2]
        # d2g = np.linalg.norm(curr_pos - self.goals[self.curr_goal]) # distance to current goal
        # reward = -1 # time penalty
        # if d2g < self.tol:
        #     reward += 50
        #     self.curr_goal += 1
        # done = (self.curr_goal == len(self.goals)) # If all goals are reached

        # new_pos = new_obs[:2]
        # if not done:
        #     self.state = np.concatenate([new_obs, self.goals[self.curr_goal]], axis=0)
        # else:
        #     self.state = np.concatenate([new_obs, self.goals[self.curr_goal-1]], axis=0)
        self.state = deepcopy(new_obs)
        self.goal = np.array(self.lower_env.goal())
        self.state = np.concatenate([self.state, self.goal], axis=0)
        
        info = {}
        self.t += 1
        all_done = all_done or (self.t == self.max_steps)

        return self.state, accumulated_reward, all_done, info

    def render(self, mode='human', close=False, message=None):
        self.lower_env.render()