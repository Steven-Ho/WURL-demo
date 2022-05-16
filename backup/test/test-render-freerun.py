import argparse
import torch
import numpy as np 
import os
import itertools
import gym
import cv2
import json
from functools import reduce
from copy import deepcopy
from gym_mm.envs.freerun import FreeRun
from argparse import Namespace
from config import parser_test
from algo.utils import wrapped_obs

customed_args = vars(parser_test().parse_args())
with open("common.json") as f:
    json_config = json.load(f)
args_dict = {}
args_dict.update(json_config["test"])
args_dict.update(customed_args)
args = Namespace(**args_dict)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Load environment
env = gym.make(args.scenario)
env.seed(None)
obs_shape_list = env.observation_space.shape
obs_shape = reduce((lambda x,y: x*y), obs_shape_list)
action_space = env.action_space
discrete_action = hasattr(action_space, 'n')

prefix = args.prefix
assert args.algo == "sac"
assert not discrete_action
args_dict.update(json_config["sac"])
args = Namespace(**args_dict)

from algo.sac import SACTrainer
if args.backend in ["diayn", "dads", "diayn_indie"]:
    args_dict.update(json_config[args.backend])
    args = Namespace(**args_dict)
    if args.backend == "diayn_indie":
        trainers = [SACTrainer(obs_shape, action_space, args) for _ in range(args.num_modes)]
        for x in range(len(trainers)):
            trainers[x].load_model(prefix+"sac_actor_{}_{}_{}".format(args.scenario, args.backend, x), prefix+"sac_critic_{}_{}_{}".format(args.scenario, args.backend, x))     
    else:
        trainers = SACTrainer(obs_shape + args.num_modes, action_space, args)
        trainers.load_model(prefix+"sac_actor_{}_{}".format(args.scenario, args.backend), prefix+"sac_critic_{}_{}".format(args.scenario, args.backend))
else:
    trainers = [SACTrainer(obs_shape, action_space, args) for _ in range(args.num_modes)]
    for x in range(len(trainers)):
        trainers[x].load_model(prefix+"sac_actor_{}_wasserstein_{}_{}".format(args.scenario, args.backend, x), prefix+"sac_critic_{}_wasserstein_{}_{}".format(args.scenario, args.backend, x))

# Initialize discriminator trainer
img = np.ones((1024, 1024, 3), np.uint8)*255
color_theme = [(152, 225, 204), (152, 213, 172), (151, 201, 137), (201, 204, 132), (244, 206, 126), 
    (218, 171, 136), (203, 151, 140), (188, 130, 143), (189, 137, 170), (191, 144, 196),
    (194, 158, 241), (159, 156, 242)]
scale_f = 51.2
boundaries = 10.0
for l in range(0, args.num_modes):
    label = np.array([l])
    for s in range(args.test_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_steps = 0
        # print("Test episode: {}".format(s))   
        trajectory = [obs[:2]]    
        while True:
            if args.backend in ["diayn", "dads"]:
                wobs = wrapped_obs(obs, label, args.num_modes)
                action, _ = trainers.act(wobs)
                if len(action.shape) > 1:
                    action = action[0]
            else:
                action, _ = trainers[l].act(obs)
                if len(action.shape) > 1:
                    action = action[0]
                if discrete_action:
                    action = action[0]  
            new_obs, reward, done, _ = env.step(action)
            done = done or (episode_reward >= args.max_episode_len)            
            trajectory.append(new_obs[:2])
            episode_reward += reward
            episode_steps += 1

            obs = new_obs
            if done:
                break
            n = len(trajectory)
            for i in range(n-1):
                s = trajectory[i]
                t = trajectory[i+1]
                s = ((s + boundaries) * scale_f).astype(np.int16).tolist()
                t = ((t + boundaries) * scale_f).astype(np.int16).tolist()
                cv2.line(img, tuple(s), tuple(t), color=color_theme[(l-1)%12], thickness=2)        

cv2.imwrite("collection.jpg", img)  