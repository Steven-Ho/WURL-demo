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
render_interval = 1

memories = [ReplayMemory(args.buffer_limit) for _ in range(args.num_modes)]
caches = [Cache(args.buffer_limit) for _ in range(args.num_modes)]
amortized = False
if args.sr_algo == 'apwd':
    amortized = True
    from algo.wsre import wsre  
elif args.sr_algo == 'wgan':
    from algo.wde import wgan
    dtrainer = wgan(obs_shape, 0, max_iter=1)
elif args.sr_algo == 'bgrl':
    from algo.wde import bgrl
    dtrainer = bgrl(obs_shape, 0, max_iter=1)

# TensorboardX
# logdir = 'logs/wasserstein_{}_{}_{}'.format(args.algo, args.scenario, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
# writer = SummaryWriter(logdir=logdir)

run = args.run
src = args.reward_scale
rc = 0
print("Reward scale for pseudo reward: {}".format(src))
print("Reward scale for external reward: {}".format(rc))
running_reward = 0
running_sr = 0
avg_length = 0
timestep = 0
updates = 0
for i_episode in itertools.count(1):
    obs = env.reset()
    episode_reward = 0
    episode_sr = 0
    trajectory = []
    episode_steps = 0
    if args.schedule == "random":
        label = np.random.randint(0, high=args.num_modes)
    else:
        p = 2 # period for each mode
        label = (i_episode//p) % args.num_modes
    for t in range(args.max_episode_len):
        if timestep < args.start_steps:
            action = env.action_space.sample()
            logprob = np.array([1.0])
        else:
            action, logprob = trainers[label].act(obs)
            if len(action.shape) > 1:
                action = action[0]
            if discrete_action:
                action = action[0]
        
        if timestep > args.start_steps:
            if args.sr_algo in ['wgan', 'bgrl']:
                target_state_batch = list(caches[1].dump(batch_size=args.disc_batch_size))[0]
                state_batch = list(caches[0].dump(batch_size=args.disc_batch_size))[0]
                d_loss = dtrainer.update_parameters(state_batch, target_state_batch)
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

        if args.sr_algo in ['wgan', 'bgrl']:
            if timestep > args.start_steps:
                sr = dtrainer.score(new_obs, label)
            else:
                sr = 0.
            memories[label].push((obs, action, logprob, sr * src + reward * rc, new_obs, mask))
            episode_sr += sr * src
        else:
            if done:
                if timestep > args.start_steps:
                    target_state_batch = []
                    for i in range(args.num_modes):
                        if i!=label:
                            s = list(caches[i].dump(episode_steps))[0]
                            target_state_batch.append(s)
                    target_state_batch = np.concatenate(target_state_batch)
                    state_batch = list(caches[label].dump(episode_steps))[0]
                    srs = wsre(state_batch, target_state_batch)    
                else:
                    srs = np.zeros(episode_steps)
                for i in range(episode_steps):
                    record = trajectory[i]
                    record[3] = record[3]* rc + srs[i] * src # Add pseudo reward to plain reward
                    memories[label].push(tuple(record))  
                episode_sr += np.sum(srs)          
  
        caches[label].push((obs,))
        episode_reward += reward
        timestep += 1

        obs = new_obs
        if done:
            break
    
    avg_length += (t+1)
    running_reward += episode_reward
    running_sr += episode_sr
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
    episode_reward = 0
    # env.render(message=str(label))
    
    if i_episode > args.num_episodes:
        break

for x in range(len(trainers)):
    trainers[x].save_model(args.scenario, prefix="models/run{}/".format(run), suffix="wasserstein_{}_{}".format(args.sr_algo, x))
env.close()         