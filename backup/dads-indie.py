import argparse
import torch
import os
import random
import numpy as np
import itertools
import datetime
import gym
import json
import mujoco_py
from functools import reduce
from argparse import Namespace
from copy import deepcopy
from tensorboardX import SummaryWriter

import exp_utils
from gym_mm.envs.freerun import FreeRun
from algo.buffer import ReplayMemory
from algo.disc import PredTrainer
from algo.utils import convert_to_onehot, wrapped_obs
from config import parser_main


start_time, exp_seed = exp_utils.get_start_time()

# load args
customed_args = vars(parser.parse_args())

with open("mujoco_configs/train/{}.json".format(customed_args["scenario"])) as f:
    json_config = json.load(f)

args_dict = {}
args_dict.update(json_config["common"])
args_dict.update(json_config[customed_args["algo"]])
args_dict.update(json_config["dads_indie"])
args_dict.update(customed_args)
args = Namespace(**args_dict)
args.seed = exp_seed
print(args)
print("now is running: {}!".format(args.run))

# args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)


if args.scenario == "AntCustom-v0":
    from gym_mm.envs.ant_custom_env import AntCustomEnv
    env = AntCustomEnv()
    reduced_obs = True
else:
    env = gym.make(args.scenario)
    reduced_obs = False
env.seed(args.seed)
obs_shape_list = env.observation_space.shape
obs_shape = reduce((lambda x,y: x*y), obs_shape_list)
action_space = env.action_space
discrete_action = hasattr(action_space, 'n')

from algo.sac import SACTrainer
trainers = [SACTrainer(obs_shape, action_space, args) for _ in range(args.num_modes)]
update_interval = 1
updates_per_step = 1
memories = [ReplayMemory(args.buffer_limit) for _ in range(args.num_modes)]
cache = ReplayMemory(args.buffer_limit)

# TensorboardX
logdir = 'logs/{}/dads_indie/dads_{}_{}_run{}_{}'.format(args.scenario, args.algo, args.scenario, args.run, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
writer = SummaryWriter(logdir=logdir)
save_args(json_config, args, logdir)

if reduced_obs:
    dtrainer = PredTrainer(2, args)
else:
    dtrainer = PredTrainer(obs_shape, args)

src = 1
rc = 0
running_reward = 0
running_sr = 0
avg_length = 0
timestep = 0
updates = 0
scale = args.reward_scale
max_mean_sr = 0
input_amp = 100
for i_episode in itertools.count(1):
    obs = env.reset()
    episode_reward = 0
    episode_sr = 0
    episode_steps = 0
    label = np.random.randint(0, high=args.num_modes)
    l = np.array([label])
    for t in range(args.max_episode_len):
        if args.start_steps < timestep:
            action, logprob = trainers[label].act(obs)
            if len(action.shape) > 1:
                action = action[0]
            if discrete_action:
                action = action[0]
        else:
            action = env.action_space.sample()
            logprob = np.array([1.0])

        if timestep > args.start_steps:
            if timestep % update_interval == 0:
                for _ in range(args.updates_per_step):
                    label_batch, state_batch, action_batch, logprob_batch, reward_batch, next_state_batch, mask_batch = memories[label].sample(batch_size=args.batch_size)

                    c1_loss, c2_loss, p_loss, ent_loss, alpha = trainers[label].update_parameters((state_batch, action_batch, logprob_batch, reward_batch, next_state_batch, mask_batch), updates)
                    writer.add_scalar('loss/critic_1', c1_loss, updates)
                    writer.add_scalar('loss/critic_2', c2_loss, updates)
                    writer.add_scalar('loss/policy', p_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)

                    updates += 1
            label_batch, state_batch, next_state_batch = cache.sample(batch_size=args.disc_batch_size)
            state_delta_batch = next_state_batch - state_batch
            label_onehot_batch = convert_to_onehot(label_batch, args.num_modes)
            if reduced_obs:
                d_loss = dtrainer.update_parameters((state_batch[:,:2], label_onehot_batch, state_delta_batch[:,:2] * input_amp))
            else:
                d_loss = dtrainer.update_parameters((state_batch, label_onehot_batch, state_delta_batch * input_amp))
            writer.add_scalar('loss/disc', d_loss, timestep)

        new_obs, reward, done, _ = env.step(action.tolist())

        if timestep > args.start_steps:
            L = args.num_modes
            alt_labels = np.concatenate([np.arange(0, label), np.arange(label+1, L)])
            obs_delta = new_obs - obs
            if reduced_obs:
                logp = dtrainer.score(obs[:2], convert_to_onehot(l, args.num_modes), obs_delta[:2] * input_amp)
            else:
                logp = dtrainer.score(obs, convert_to_onehot(l, args.num_modes), obs_delta * input_amp)
            alt_obs = np.tile(obs, [L-1,1])
            alt_new_obs = np.tile(new_obs, [L-1,1])
            alt_obs_delta = alt_new_obs - alt_obs
            if reduced_obs:
                alt_logp = dtrainer.score(alt_obs[:,:2], convert_to_onehot(alt_labels, args.num_modes), alt_obs_delta[:,:2] * input_amp)
            else:
                alt_logp = dtrainer.score(alt_obs, convert_to_onehot(alt_labels, args.num_modes), alt_obs_delta * input_amp)

            writer.add_scalar('logp/logp', logp, timestep)
            writer.add_scalar('logp/alt_logp', alt_logp.mean(), timestep)
            writer.add_scalar('logp/alt_logp_max', alt_logp.max(), timestep)
            writer.add_scalar('bn/var', dtrainer.pred.output_bn.running_var.mean().item(), timestep)
            sr = np.log(L) - np.log(1+np.exp(np.clip(alt_logp - logp, -20, 1)).sum(axis=0))
        else:
            sr = 0.
        sr *= scale

        episode_reward += reward
        episode_sr += sr
        timestep += 1
        episode_steps += 1
        if hasattr(env, 'max_steps'):
            mask = 1 if episode_steps == env.max_steps else float(not done)
        else:
            mask = float(not done)

        memories[label].push((label, obs, action, logprob, sr * src + reward * rc, new_obs, mask))
        cache.push((l, obs, new_obs))
        obs = new_obs
        if done:
            break

    avg_length += (t+1)
    running_reward += episode_reward
    running_sr += episode_sr
    # env.render(message=str(label))
    writer.add_scalar('stats/episode_reward', episode_reward, i_episode)
    writer.add_scalar('stats/episode_sr', episode_sr, i_episode)
    writer.add_scalar('stats/episode_length', t, i_episode)

    if i_episode % args.log_interval == 0:
        print("Episode: {}, length: {}, reward: {}, sr: {}".format(i_episode, int(avg_length/args.log_interval),
            int(running_reward/args.log_interval), int(running_sr/args.log_interval)))
        if running_sr/args.log_interval > max_mean_sr:
            max_mean_sr = running_sr/args.log_interval
            for x in range(len(trainers)):
                trainers[x].save_model(args.scenario, prefix="models/{}/dads_indie/run{}/".format(args.scenario, args.run), suffix="dads_indie_{}".format(x), silent=True)
            print("Models saved to "+"models/{}/dads_indie/run{}".format(args.scenario, args.run))

        avg_length = 0
        running_reward = 0
        running_sr = 0
    episode_reward = 0

    if i_episode > args.num_episodes:
        break

# trainer.save_model(args.scenario, suffix="dads")
# print("Models saved to "+"models/{}/run{}/".format(args.scenario, args.run))
env.close()

now_time = datetime.datetime.now().strftime("%Y.%m.%d - %H:%M:%S")
exp_time = "{} -> {}".format(start_time, now_time)
print("This experiment time: {}".format(exp_time))

exp_utils.save_exp_config(args, exp_time, exp_name="dads-indie")
