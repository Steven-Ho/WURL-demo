import argparse
import torch
import numpy as np
import os
import itertools
import gym
import cv2
import json
import mujoco_py
from functools import reduce
from copy import deepcopy
from argparse import Namespace
from mujoco_maze import PointEnv
from config import parser_test
from algo.utils import wrapped_obs

customed_args = vars(parser_test().parse_args())
with open("mujoco_configs/{}.json".format(customed_args['scenario'])) as f:
    json_config = json.load(f)

args_dict = {}
args_dict.update(customed_args)
args_dict.update(json_config["test"])
args = Namespace(**args_dict)
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Load environment
env = gym.make(args.scenario) if args.scenario != "Ant-v3" else gym.make(
    args.scenario,
    exclude_current_positions_from_observation=args.exclude_current_positions_from_observation
)  # add xy position in observation for Ant-v3
env.seed(args.seed)
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
if args.backend in ["diayn", "dads"]:
    args_dict.update(json_config[args.backend])
    args = Namespace(**args_dict)
    trainers = SACTrainer(obs_shape + args.num_modes, action_space, args)
    trainers.load_model(prefix+"sac_actor_{}_{}".format(args.scenario, args.backend), prefix+"sac_critic_{}_{}".format(args.scenario, args.backend))
else:
    trainers = [SACTrainer(obs_shape, action_space, args) for _ in range(args.num_modes)]
    for x in range(len(trainers)):
        trainers[x].load_model(prefix+"sac_actor_{}_wasserstein_{}_{}".format(args.scenario, args.backend, x), prefix+"sac_critic_{}_wasserstein_{}_{}".format(args.scenario, args.backend, x))

for l in range(1, args.num_modes):
    label = np.array([l])
    for s in range(2):
        obs = env.reset()
        episode_reward = 0
        episode_steps = 0
        print('----------------------------------------------')
        print("Mode: {}, Test episode: {}".format(l, s))
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
            new_obs, reward, done, info = env.step(action)
            done = done or (episode_steps >= args.max_episode_len)
            episode_reward += reward
            episode_steps += 1

            obs = new_obs
            env.render()
            if done:
                break
        print('episode length', episode_steps)
        print('final xy position: {:.5f}, {:.5f}'.format(info['x_position'], info['y_position']))
