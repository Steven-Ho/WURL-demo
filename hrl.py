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
print("Sub args: {}".format(sub_args))

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
memory = ReplayMemory(args.buffer_limit)
update_interval = 1000
updates_per_step = 1

# TensorboardX
logdir = 'logs/hrl/AntCustom/{}/skill{}/run{}_{}_{}_{}'.format(args.backend, args.skill_run, args.run, args.backend, args.scenario, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
writer = SummaryWriter(logdir=logdir)

running_reward = 0
avg_length = 0
timestep = 0
updates = 0
max_mean_reward = -50
for i_episode in itertools.count(1):
    obs = env.reset()
    episode_reward = 0
    episode_steps = 0
    for t in range(args.max_episode_len):
        if args.start_steps < timestep:
            action, logprob = trainer.act(obs)
            if len(action.shape) > 1:
                action = action[0]
            if discrete_action:
                action = action[0]
        else:
            action = env.action_space.sample()
            logprob = np.array([1.0])
        # action = np.random.randint(0, args.num_modes, size=(1,))

        if timestep > args.start_steps:
            if timestep % update_interval == 0:
                for _ in range(args.updates_per_step):
                    # state_batch, action_batch, logprob_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=args.batch_size)
                    state_batch, action_batch, logprob_batch, reward_batch, next_state_batch, mask_batch = memory.dump(1000)

                    # c1_loss, c2_loss, p_loss, ent_loss, alpha = trainer.update_parameters((state_batch, action_batch, logprob_batch, reward_batch, next_state_batch, mask_batch), updates)
                    # writer.add_scalar('loss/critic_1', c1_loss, updates)
                    # writer.add_scalar('loss/critic_2', c2_loss, updates)
                    # writer.add_scalar('loss/policy', p_loss, updates)
                    # writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    # writer.add_scalar('entropy_temprature/alpha', alpha, updates)                        
                    lp, lq, le = trainer.update_parameters((state_batch, action_batch, logprob_batch, reward_batch, next_state_batch, mask_batch))
                    writer.add_scalar('loss/critic', lq, updates)
                    writer.add_scalar('loss/policy', lp, updates)
                    writer.add_scalar('loss/entropy_loss', le, updates)                     
                    updates += 1

        new_obs, reward, done, _ = env.step(action)
        episode_reward += reward
        timestep += 1
        episode_steps += 1
        if hasattr(env, 'max_steps'):
            mask = 1 if episode_steps == env.max_steps else float(not done)
        else:
            mask = float(not done)

        memory.push((obs, action, logprob, reward, new_obs, mask))

        obs = new_obs
        # env.render()
        if done:
            break

    avg_length += (t+1)
    running_reward += episode_reward
    # env.render()
    writer.add_scalar('stats/episode_reward', episode_reward, i_episode)
    writer.add_scalar('stats/episode_length', t, i_episode)

    if i_episode % args.log_interval == 0:
        print("Episode: {}, length: {}, reward: {}".format(i_episode, int(avg_length/args.log_interval), 
            int(running_reward/args.log_interval)))
            # trainer.save_model(args.scenario, prefix="models/{}/run{}/".format(args.scenario, run), suffix="diayn_indie_{}".format(x), silent=True)
        if running_reward/args.log_interval > max_mean_reward:
            trainer.save_model(args.scenario, prefix="models/hrl/{}/subskill{}/run{}/".format(args.backend, args.skill_run, args.run), suffix="{}".format(args.backend), silent=True)
            max_mean_reward = running_reward/args.log_interval
        # env.render()
        avg_length = 0
        running_reward = 0
        running_sr = 0
    episode_reward = 0
    if i_episode > args.num_episodes:
        break

env.close()

now_time = datetime.datetime.now().strftime("%Y.%m.%d - %H:%M:%S")
exp_time = "{} -> {}".format(start_time, now_time)


def save_hrl_logging(args, exp_time):
    dir_path = "results/hrl/{}/{}".format(args.scenario, args.backend)

    # read data from csv
    try:
        df = pd.read_csv(os.path.join(dir_path, "skill{}-run{}.csv".format(args.skill_run, args.run)), index_col=0)
    except:
        df = pd.DataFrame(columns=["Skill ID", "Run ID", "Reward Mean", "Reward Std", "Train Seed", "Train Time", "Test Seed", "Test Time"])

    # add new data
    df.loc[df.shape[0]] = [args.skill_run, args.run, None, None, args.seed, exp_time, None, None]

    # save
    exp_utils.assert_path(dir_path)
    df.to_csv(os.path.join(dir_path, "skill{}-run{}.csv".format(args.skill_run, args.run)))


save_hrl_logging(args, exp_time)

print("This experiment time: {}".format(exp_time))
print("------------------------------------------------------------------------------------")
