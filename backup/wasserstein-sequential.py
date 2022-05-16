import time
import datetime
import torch
import numpy as np 
import os
import itertools
import gym
import json
from functools import reduce
from copy import deepcopy
from tensorboardX import SummaryWriter
from gym_mm.envs.freerun import FreeRun
from gym_mm.envs.freerund import FreeRunD
from mujoco_maze import PointEnv
from algo.buffer import ReplayMemory, Cache
from config import parser_main
from argparse import Namespace

customed_args = vars(parser_main().parse_args())
with open("common.json") as f:
    json_config = json.load(f)
args_dict = {}
args_dict.update(json_config["common"])
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

if args.algo == "sac":
    args_dict.update(json_config["sac"])
    args = Namespace(**args_dict)
    if discrete_action:
        from algo.sacd import SACDTrainer
        trainers = [SACDTrainer(obs_shape, action_space, args) for _ in range(args.num_modes)]
    else:
        from algo.sac import SACTrainer
        trainers = [SACTrainer(obs_shape, action_space, args) for _ in range(args.num_modes)]
    update_interval = 1
    updates_per_step = 1
else: # PPO
    args_dict.update(json_config["ppo"])
    args = Namespace(**args_dict)
    if discrete_action:
        from algo.ppo import PPOTrainer
        trainers = [PPOTrainer(obs_shape, action_space.n, args) for _ in range(args.num_modes)]
    else:
        from algo.ppoc import PPOCTrainer
        trainers = [PPOCTrainer(obs_shape, action_space.shape[0], args) for _ in range(args.num_modes)]
    update_interval = 1000
    updates_per_step = 5

memories = [ReplayMemory(args.buffer_limit) for _ in range(args.num_modes)]
caches = [Cache(args.buffer_limit) for _ in range(args.num_modes)]

amortized = False
if args.sr_algo not in ["apwd", "pwil", "pwd"]:
    raise ValueError("Only primal WDE methods are supported!")
elif args.sr_algo == "apwd":
    amortized = True
    from algo.wsre import wsre
elif args.sr_algo == "pwil":
    from algo.wde import pwil
    wde = pwil()
elif args.sr_algo == "pwd":
    from algo.wde import apwd
    wde = apwd()

# TensorboardX
# logdir = 'logs/wasserstein_{}_{}_{}'.format(args.algo, args.scenario, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
# writer = SummaryWriter(logdir=logdir)

run = args.run

args.start_steps = 1000
src = args.reward_scale
rc = 0
running_reward = 0
running_sr = 0
avg_length = 0
timestep = 0
updates = 0
episodes_per_mode = 200
for i_episode in itertools.count(1):
    obs = env.reset()
    episode_reward = 0
    episode_sr = 0
    trajectory = []
    episode_steps = 0
    label = i_episode // episodes_per_mode
    if label >= args.num_modes:
        break
    while True:
        if label == 0:
            #action = env.action_space.sample()
            action = np.array([0.0, 0.0])
            logprob = np.array([1.0])
        else:
            action, logprob = trainers[label].act(obs)
            if len(action.shape) > 1:
                action = action[0]
            if discrete_action:
                action = action[0]
        
        if len(memories[label]) > args.start_steps and label > 0:
            if timestep % update_interval == 0:
                for _ in range(updates_per_step):
                    if args.algo == "sac":                      
                        c1_loss, c2_loss, p_loss, ent_loss, alpha = trainers[label].update_parameters(memories[label].sample(batch_size=args.batch_size), updates)
                        # writer.add_scalar('loss/critic_1', c1_loss, updates)
                        # writer.add_scalar('loss/critic_2', c2_loss, updates)
                        # writer.add_scalar('loss/policy', p_loss, updates)
                        # writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                        # writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    else:
                        lp, lq, le = trainers[label].update_parameters(memories[label].dump(2000))
                        # writer.add_scalar('loss/critic', lq, updates)
                        # writer.add_scalar('loss/policy', lp, updates)
                        # writer.add_scalar('loss/entropy_loss', le, updates)                           
                    updates += 1                        

        if type(action) is np.ndarray:
            action = action.tolist()
        new_obs, reward, done, _ = env.step(action)

        if hasattr(env, 'max_steps'):
            mask = 1 if episode_steps == env.max_steps else float(not done)
        else:
            mask = float(not done)
        done = done or (episode_steps >= args.max_episode_len)
        trajectory.append([obs, action, logprob, reward, new_obs, mask])
        episode_steps += 1

        bc = args.max_episode_len
        if done:
            srs = np.zeros(episode_steps)
            if len(memories[label]) > args.start_steps and label > 0:
                state_batch = list(caches[label].dump(episode_steps))[0]
                srs_list = []
                sum_srs = []
                for i in range(label):
                    target_state_batch = list(caches[i].dump(episode_steps))[0]  
                    if amortized:          
                        sr = wsre(state_batch, target_state_batch) 
                        sum_srs.append(np.sum(sr))
                        srs_list.append(sr)
                    else:
                        sr = wde.estimate(state_batch, target_state_batch)
                        srs_list.append(sr)
                if amortized:
                    min_dist_idx = np.argmin(sum_srs) # find the nearest policy
                    srs = srs_list[min_dist_idx]
                    # srs_list = np.stack(srs_list)
                    # srs = np.mean(srs_list, axis=0)
                else:
                    srs[-1] = min(srs_list)
                    # srs[-1] = np.mean(srs_list)

            for i in range(episode_steps):
                record = trajectory[i]
                record[3] = record[3]* rc + srs[i] * src * (bc / episode_steps)# Add pseudo reward to plain reward
                memories[label].push(tuple(record))  
            episode_sr += np.sum(srs) * src          
  
        caches[label].push((obs,))
        episode_reward += reward
        timestep += 1

        obs = new_obs
        # env.render()
        if done:
            break
    
    avg_length += (episode_steps-1)
    running_reward += episode_reward
    running_sr += episode_sr
    # env.render(message=str(label))
    # env.render()
    # writer.add_scalar('stats/episode_reward', episode_reward, i_episode)
    # writer.add_scalar('stats/episode_sr', episode_sr, i_episode)
    # writer.add_scalar('stats/episode_length', t, i_episode)

    if i_episode % args.log_interval == 0:
        print("Episode: {}, length: {}, reward: {}, sr: {}".format(i_episode, int(avg_length/args.log_interval), 
            int(running_reward/args.log_interval), int(running_sr/args.log_interval)))
        avg_length = 0
        running_reward = 0
        running_sr = 0
        # periodically save models
        trainers[label].save_model(args.scenario, prefix="models/{}/seq{}/".format(args.scenario, run), suffix="wasserstein_{}_{}".format(args.sr_algo, label), silent=True)
    episode_reward = 0
    
    # if i_episode > args.num_episodes:
    #     break

for x in range(len(trainers)):
    trainers[x].save_model(args.scenario, prefix="models/{}/seq{}/".format(args.scenario, run), suffix="wasserstein_{}_{}".format(args.sr_algo, x))
env.close()         