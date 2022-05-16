import argparse
import time
import datetime
import torch
import numpy as np 
import os
import itertools
import gym
import mujoco_py
import json
import pandas as pd
from tensorboardX import SummaryWriter

import exp_utils
from config import parser_hrl
from gym_mm.envs.navi_cont import Navi2DCont
from argparse import Namespace
from functools import reduce
from algo.buffer import ReplayMemory
from algo.sacd import SACDTrainer
from algo.ppo import PPOTrainer
from gym_mm.envs.ant_custom_env import AntCustomEnv
from gym_mm.envs.ant_skill import AntSkill


print("------------------------------------------------------------------------------------")
start_time, exp_seed = exp_utils.get_start_time()

# load args
customed_args = vars(parser_hrl().parse_args())

with open("mujoco_configs/train/AntCustom-v0.json") as f:
    json_config = json.load(f)

args_dict = {}
args_dict.update(json_config["meta"])
args_dict.update(json_config["ppo"])
args_dict.update(customed_args)
args = Namespace(**args_dict)
args.seed = exp_seed
print("Meta args: {}".format(args))

sub_args_dict = {"backend": args.backend, "prefix": args.prefix}
sub_args_dict.update(json_config["sub"])
sub_args_dict.update(json_config["sac"])

if args.backend in ["diayn", "dads"]:
    sub_args_dict.update(json_config[args.backend])
sub_args_dict.update({"num_modes": args.num_modes,
                       "seed": args.seed,
                       "prefix": args.prefix})

sub_args = Namespace(**sub_args_dict)

# create env
reduced_obs = False
if args.scenario == "AntCustom-v0":
    reduced_obs = True
    env = AntSkill(sub_args)
else:
    env = gym.make("FreeRunSkill-v0", args=sub_args)

exp_utils.setup_seed(args.seed)
env.seed(args.seed)

obs_shape_list = env.observation_space.shape
obs_shape = reduce((lambda x,y: x*y), obs_shape_list)
action_space = env.action_space
discrete_action = hasattr(action_space, 'n')

trainer = PPOTrainer(obs_shape, action_space.n, args)
trainer.load_model("models/hrl/{}/subskill{}/run{}/ppo_{}_{}".format(args.backend, args.skill_run, args.run, args.scenario, args.backend))
rewards = []
for i in range(10):
    obs = env.reset()
    episode_reward = 0
    episode_steps = 0
    for t in range(args.max_episode_len):
        # action, logprob = trainer.act(obs, eval=False)
        action, logprob = trainer.act(obs)
        if len(action.shape) > 1:
            action = action[0]
        if discrete_action:
            action = action[0]

        new_obs, reward, done, _ = env.step(action)
        episode_reward += reward
        episode_steps += 1

        obs = new_obs
        if done:
            break
    rewards.append(episode_reward)

env.close()

now_time = datetime.datetime.now().strftime("%Y.%m.%d - %H:%M:%S")
exp_time = "{} -> {}".format(start_time, now_time)

print("This test time: {}".format(exp_time))

reward_mean = np.mean(rewards)
reward_std = np.std(rewards)


def save_hrl_result(args, exp_time, reward_mean, reward_std):
    dir_path = "results/hrl/{}/{}".format(args.scenario, args.backend)

    # read data from csv

    df = pd.read_csv(os.path.join(dir_path, "skill{}-run{}.csv".format(args.skill_run, args.run)), index_col=0)

    # add new data
    df.loc[0, "Reward Mean"] = reward_mean
    df.loc[0, "Reward Std"] = reward_std
    df.loc[0, "Test Seed"] = args.seed
    df.loc[0, "Test Time"] = exp_time

    # save
    df.to_csv(os.path.join(dir_path, "skill{}-run{}.csv".format(args.skill_run, args.run)))

save_hrl_result(args, exp_time, reward_mean, reward_std)
print("Reward mean: {}, std: {}".format(reward_mean, reward_std))
print("------------------------------------------------------------------------------------")
