import argparse
import torch
import datetime
import random
import numpy as np
import os
import itertools
import gym
import json
import mujoco_py
from argparse import Namespace
from functools import reduce
from copy import deepcopy

import exp_utils
from gym_mm.envs.freerun import FreeRun
from gym_mm.envs.freerund import FreeRunD
from mujoco_maze import PointEnv
from algo.buffer import ReplayMemory, Cache
from algo.disc import DiscTrainer
from algo.utils import wrapped_obs
from config import parser_test


print("------------------------------------------------------------------------------------")
start_time, exp_seed = exp_utils.get_start_time()

# load args
customed_args = vars(parser_test().parse_args())

if customed_args["scenario"] in ["HalfCheetah-v3", "AntCustom-v0", "Humanoid-v3"]:
    mujoco_env = True
    with open("mujoco_configs/test/{}.json".format(customed_args["scenario"])) as f:
        json_config = json.load(f)
else:
    with open("common.json") as f:
        json_config = json.load(f)

args_dict = {}
args_dict.update(json_config["test"])
assert customed_args["algo"]=="sac"
args_dict.update(json_config["sac"])
args_dict.update(json_config[customed_args["backend"]])
args_dict.update(customed_args)
args = Namespace(**args_dict)
args.seed = exp_seed
print(args)
print("now is testing: {} - skill{}".format(args.backend, args.skill_run))

# create env
if args.scenario == "AntCustom-v0":
    from gym_mm.envs.ant_custom_env import AntCustomEnv
    env = AntCustomEnv()
    reduced_obs = True
else:
    env = gym.make(args.scenario)
    reduced_obs = False

exp_utils.setup_seed(args.seed)
env.seed(args.seed)

obs_shape_list = env.observation_space.shape
obs_shape = reduce((lambda x,y: x*y), obs_shape_list)
action_space = env.action_space
discrete_action = hasattr(action_space, 'n')

prefix = args.prefix + args.skill_run + "/"
assert args.algo == "sac"
assert not discrete_action

from algo.sac import SACTrainer
if args.backend in ["diayn", "dads"]:
    trainers = [SACTrainer(obs_shape, action_space, args) for _ in range(args.num_modes)]
    for x in range(len(trainers)):
        trainers[x].load_model(prefix+"sac_actor_{}_{}_{}".format(args.scenario, args.backend, x), prefix+"sac_critic_{}_{}_{}".format(args.scenario, args.backend, x))
else:
    trainers = [SACTrainer(obs_shape, action_space, args) for _ in range(args.num_modes)]
    for x in range(len(trainers)):
        trainers[x].load_model(prefix+"sac_actor_{}_wurl_{}_{}".format(args.scenario, args.backend, x), prefix+"sac_critic_{}_wasserstein_{}_{}".format(args.scenario, args.backend, x))

# Initialize discriminator trainer
dtrainer = DiscTrainer(obs_shape, args)
mixed_buffer = ReplayMemory(args.buffer_limit)
indie_buffer = [Cache(args.buffer_limit) for _ in range(args.num_modes)]
for l in range(args.num_modes):
    label = np.array([l])
    for s in range(args.test_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_steps = 0
        # print("---------------------------------")
        # print("Mode:{} Test episode: {}".format(l, s))
        # while True:
        for t in range(args.max_episode_len):
            # remove action sample process for vanilla diayn and dads here.
            # if args.backend in ["diayn", "dads"]:
            #     wobs = wrapped_obs(obs, label, args.num_modes)
            #     action, _ = trainers.act(wobs)
            #     if len(action.shape) > 1:
            #         action = action[0]
            # else:
            action, _ = trainers[l].act(obs)
            if len(action.shape) > 1:
                action = action[0]
            if discrete_action:
                action = action[0]

            new_obs, reward, done, _ = env.step(action)
            mixed_buffer.push((label, new_obs))
            indie_buffer[l].push((new_obs,))
            done = done or (episode_reward >= args.max_episode_len)

            episode_reward += reward
            episode_steps += 1
            obs = new_obs
            # env.render()
            if done:
                break
        # print("    episode reward: {}, episode steps: {}".format(episode_reward, episode_steps))

# Evaluation
# Neural discriminator
for _ in range(1000):
    label_batch, state_batch = mixed_buffer.sample(200)
    loss = dtrainer.update_parameters((label_batch, state_batch))
label_batch, state_batch = mixed_buffer.dump(len(mixed_buffer))
scores = dtrainer.score(state_batch, label_batch)
disc_score = np.exp(np.mean(scores))

print("------------------------------------------------")
print("{}\tskill{}".format(args.backend, args.skill_run))
print("The discriminator score is:", disc_score)

# Wasserstein distance estimator
if args.wde_algo == 'apwd':
    from algo.wsre import wsre
elif args.wde_algo == 'wgan':
    from algo.wde import wgan
    dtrainer = wgan(obs_shape, 0, max_iter=1000)
elif args.wde_algo == 'bgrl':
    from algo.wde import bgrl
    dtrainer = bgrl(obs_shape, 0, max_iter=1000)

scores = []
for i in range(args.num_modes):
    state_batch = list(indie_buffer[i].dump(len(indie_buffer[i])))[0]
    wds = []
    for j in range(args.num_modes):
        if j == i:
            continue
        target_state_batch = list(indie_buffer[j].dump(len(indie_buffer[j])))[0]
        if args.wde_algo == 'apwd':
            score = np.mean(wsre(state_batch, target_state_batch))
        else:
            dtrainer.train(state_batch, target_state_batch)
            score = dtrainer.estimate(state_batch, target_state_batch)
            score = abs(score)
        wds.append(score)
    scores.append(np.min(wds))
wd_score = np.mean(scores)

print("The Wasserstein Distance score is:", wd_score)

now_time = datetime.datetime.now().strftime("%Y.%m.%d - %H:%M:%S")
exp_time = "{} -> {}".format(start_time, now_time)
print("This test time is: {}".format(exp_time))
print("------------------------------------------------")

exp_utils.save_exp_result(disc_score, wd_score, args, exp_time, exp_name="wurl" if args.backend=="apwd" else args.backend, idx=0)
