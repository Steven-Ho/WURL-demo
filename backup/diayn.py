import argparse
import time
import datetime
import torch
import numpy as np
import os
import itertools
import gym
import json
import mujoco_py
from functools import reduce
from copy import deepcopy
from gym_mm.envs.navi_cont import Navi2DCont
from mujoco_maze import PointEnv
from algo.buffer import ReplayMemory
from algo.disc import DiscTrainer
from argparse import Namespace
from config import parser_main
from algo.utils import wrapped_obs
from tensorboardX import SummaryWriter


start_time = datetime.datetime.now()

customed_args = vars(parser_main().parse_args())
with open("mujoco.json") as f:
    json_config = json.load(f)

args_dict = {}
args_dict.update(customed_args)
args_dict.update(json_config["common"])
args_dict.update(json_config["diayn"])
args = Namespace(**args_dict)
print(args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Load environment
env = gym.make(args.scenario)
env.seed(None)
obs_shape_list = env.observation_space.shape
obs_shape = reduce((lambda x,y: x*y), obs_shape_list)
action_space = env.action_space
discrete_action = hasattr(action_space, 'n')

if args.algo == "sac":
    args_dict.update(json_config["sac"])
    args = Namespace(**args_dict)
    if discrete_action:
        from algo.sacd import SACDTrainer
        trainer = SACDTrainer(obs_shape + args.num_modes, action_space, args)
    else:
        from algo.sac import SACTrainer
        trainer = SACTrainer(obs_shape + args.num_modes, action_space, args)
    update_interval = 1
    updates_per_step = 1
else: # PPO
    args_dict.update(json_config["ppo"])
    args = Namespace(**args_dict)
    if discrete_action:
        from algo.ppo import PPOTrainer
        trainer = PPOTrainer(obs_shape + args.num_modes, action_space.n, args)
    else:
        from algo.ppoc import PPOCTrainer
        trainer = PPOCTrainer(obs_shape + args.num_modes, action_space.shape[0], args)
    update_interval = 1000
    updates_per_step = 5

# args.num_episodes = 1000
args.schedule = "random"
memory = ReplayMemory(args.buffer_limit)

# TensorboardX
if not os.path.exists("logs/{}".format(args.scenario)):
    os.makedirs("logs/{}".format(args.scenario))

logdir = 'logs/{}/diayn_{}_{}_run{}_{}'.format(args.scenario, args.algo, args.scenario, args.run, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
writer = SummaryWriter(logdir=logdir)

# Initialize trainer
dtrainer = DiscTrainer(obs_shape, args)
run = args.run
src = args.reward_scale
rc = 0
print("Reward scale for pseudo reward: {}".format(src))
print("Reward scale for external reward: {}".format(rc))
running_reward = 0
max_mean_sr = -233
running_sr = 0
avg_length = 0
timestep = 0
updates = 0
lower = np.log(1/args.num_modes)*10
for i_episode in itertools.count(1):
    obs = env.reset()
    episode_reward = 0
    episode_sr = 0
    episode_steps = 0
    label = np.random.randint(0, high=args.num_modes)
    label = np.array([label])
    for t in range(args.max_episode_len):
        wobs = wrapped_obs(obs, label, args.num_modes)
        if args.start_steps < timestep:
            action, logprob = trainer.act(wobs)
            if len(action.shape) > 1:
                action = action[0]
            if discrete_action:
                action = action[0]
        else:
            action = env.action_space.sample()
            logprob = np.array([1.0])
        sr = max(dtrainer.score(obs, label), lower)

        if len(memory) > args.start_steps:
            if timestep % update_interval == 0:
                for _ in range(args.updates_per_step):
                    if args.algo == "sac":
                        label_batch, state_batch, action_batch, logprob_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=args.batch_size)
                    else:
                        label_batch, state_batch, action_batch, logprob_batch, reward_batch, next_state_batch, mask_batch = memory.dump(2000)
                    wstate_batch = wrapped_obs(state_batch, label_batch, args.num_modes)
                    wnext_state_batch = wrapped_obs(next_state_batch, label_batch, args.num_modes)
                    if args.algo == "sac":
                        c1_loss, c2_loss, p_loss, ent_loss, alpha = trainer.update_parameters((wstate_batch, action_batch, logprob_batch, reward_batch, wnext_state_batch, mask_batch), updates)
                        writer.add_scalar('loss/critic_1', c1_loss, updates)
                        writer.add_scalar('loss/critic_2', c2_loss, updates)
                        writer.add_scalar('loss/policy', p_loss, updates)
                        writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                        writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    else:
                        lp, lq, le = trainer.update_parameters((wstate_batch, action_batch, logprob_batch, reward_batch, wnext_state_batch, mask_batch))
                        writer.add_scalar('loss/critic', lq, updates)
                        writer.add_scalar('loss/policy', lp, updates)
                        writer.add_scalar('loss/entropy_loss', le, updates)
                    updates += 1
            label_batch, state_batch, _, _, _, _, _ = memory.sample(batch_size=args.disc_batch_size)
            d_loss = dtrainer.update_parameters((label_batch, state_batch))
            writer.add_scalar('loss/disc', d_loss, timestep)

        new_obs, reward, done, _ = env.step(action.tolist())

        episode_reward += reward
        episode_sr += sr*src
        timestep += 1
        episode_steps += 1
        if hasattr(env, 'max_steps'):
            mask = 1 if episode_steps == env.max_steps else float(not done)
        else:
            mask = float(not done)

        memory.push((label, obs, action, logprob, sr * src + reward * rc, new_obs, mask))
        obs = new_obs
        if done:
            break

    avg_length += (t+1)
    running_reward += episode_reward
    running_sr += episode_sr
    # env.render()
    writer.add_scalar('stats/episode_reward', episode_reward, i_episode)
    writer.add_scalar('stats/episode_sr', episode_sr, i_episode)
    writer.add_scalar('stats/episode_length', t, i_episode)

    if i_episode % args.log_interval == 0:
        print("Episode: {}, length: {}, reward: {}, sr: {}".format(i_episode, int(avg_length/args.log_interval),
            int(running_reward/args.log_interval), int(running_sr/args.log_interval)))
        if running_sr/args.log_interval > max_mean_sr:
            trainer.save_model(args.scenario, prefix="models/{}/diayn/run{}/".format(args.scenario, run), suffix="diayn", silent=True)
            max_mean_sr = running_sr/args.log_interval
            print("Models saved to "+"models/{}/diayn/run{}/".format(args.scenario, run))
        avg_length = 0
        running_reward = 0
        running_sr = 0
    episode_reward = 0

    if i_episode > args.num_episodes:
        break
trainer.save_model(args.scenario, prefix="models/{}/diayn/run{}/".format(args.scenario, run), suffix="diayn")
print("Models saved to "+"models/{}/diayn/run{}/".format(args.scenario, run))
env.close()

now_time = datetime.datetime.now()
print("This experiment time: {} -> {}".format(start_time.strftime("%Y.%m.%d - %H-%M-%S"),
                                              now_time.strftime("%Y.%m.%d - %H-%M-%S")))
