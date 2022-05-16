import argparse
import time
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
from algo.buffer import ReplayMemory, Cache
from algo.disc import PredTrainer, DiscTrainer
from algo.utils import convert_to_onehot, wrapped_obs
from config import parser_train


def dads(args, env, writer, **kwargs):
    obs_shape_list = env.observation_space.shape
    obs_shape = reduce((lambda x,y: x*y), obs_shape_list)
    action_space = env.action_space
    discrete_action = hasattr(action_space, 'n')
    
    from algo.sac import SACTrainer
    trainers = [SACTrainer(obs_shape, action_space, args) for _ in range(args.num_modes)]
    args.target_update_interval = 1
    args.updates_per_step = 1
    
    memories = [ReplayMemory(args.buffer_limit) for _ in range(args.num_modes)]
    cache = ReplayMemory(args.buffer_limit)

    if reduced_obs:
        dtrainer = PredTrainer(2, args)
    else:
        dtrainer = PredTrainer(obs_shape, args)
    
    model_prefix = "models/{}/{}/run{}/".format(args.scenario, args.exp_name, args.run)
    exp_utils.assert_path(model_prefix)
    
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
                if timestep % args.target_update_interval == 0:
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
                    save_trainer(trainers[x], args, model_prefix, x)
                print("Models have been saved to {}".format(model_prefix))

            avg_length = 0
            running_reward = 0
            running_sr = 0
        episode_reward = 0

        if i_episode > args.num_episodes:
            break
    

def diayn(args, env, writer, **kwargs):
    obs_shape_list = env.observation_space.shape
    obs_shape = reduce((lambda x,y: x*y), obs_shape_list)
    action_space = env.action_space
    discrete_action = hasattr(action_space, 'n')
    
    if args.algo == "sac":
        if discrete_action:
            from algo.sacd import SACDTrainer
            trainer = SACDTrainer(obs_shape + args.num_modes, action_space, args)
        else:
            from algo.sac import SACTrainer
            trainers = [SACTrainer(obs_shape, action_space, args) for _ in range(args.num_modes)]
        args.target_update_interval = 1
        args.updates_per_step = 1
    else: # PPO
        if discrete_action:
            from algo.ppo import PPOTrainer
            trainer = PPOTrainer(obs_shape + args.num_modes, action_space.n, args)
        else:
            from algo.ppoc import PPOCTrainer
            trainer = PPOCTrainer(obs_shape + args.num_modes, action_space.shape[0], args)
        args.target_update_interval = 1000
        args.updates_per_step = 5

    memories = [ReplayMemory(args.buffer_limit) for _ in range(args.num_modes)]
    cache = ReplayMemory(args.buffer_limit)

    # Initialize trainer
    dtrainer = DiscTrainer(obs_shape, args)
    
    model_prefix = "models/{}/{}/run{}/".format(args.scenario, args.exp_name, args.run)
    exp_utils.assert_path(model_prefix)
    
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
        if args.schedule == "random" and timestep > args.start_steps:
            label = np.random.randint(0, high=args.num_modes)
        else:
            p = 2 # period for each mode
            label = (i_episode//p) % args.num_modes
        l = np.array([label])
        for t in range(args.max_episode_len):
            wobs = wrapped_obs(obs, l, args.num_modes)
            if args.start_steps < timestep:
                # action, logprob = trainer.act(wobs)
                action, logprob = trainers[label].act(obs)
                if len(action.shape) > 1:
                    action = action[0]
                if discrete_action:
                    action = action[0]
            else:
                action = env.action_space.sample()
                logprob = np.array([1.0])
            sr = max(dtrainer.score(obs, l), lower)

            if timestep > args.start_steps:
                if timestep % args.target_update_interval == 0:
                    for _ in range(args.updates_per_step):
                        if args.algo == "sac":
                            label_batch, state_batch, action_batch, logprob_batch, reward_batch, next_state_batch, mask_batch = memories[label].sample(batch_size=args.batch_size)
                        else:
                            label_batch, state_batch, action_batch, logprob_batch, reward_batch, next_state_batch, mask_batch = memories[label].dump(2000)
                        wstate_batch = wrapped_obs(state_batch, label_batch, args.num_modes)
                        wnext_state_batch = wrapped_obs(next_state_batch, label_batch, args.num_modes)
                        if args.algo == "sac":
                            c1_loss, c2_loss, p_loss, ent_loss, alpha = trainers[label].update_parameters((state_batch, action_batch, logprob_batch, reward_batch, next_state_batch, mask_batch), updates)
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
                label_batch, state_batch = cache.sample(batch_size=args.disc_batch_size)
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

            memories[label].push((l, obs, action, logprob, sr * src + reward * rc, new_obs, mask))
            cache.push((l, obs))
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
                max_mean_sr = running_sr/args.log_interval
                for x in range(len(trainers)):
                    save_trainer(trainers[x], args, model_prefix, x)
                print("Models have been saved to {}".format(model_prefix))

            avg_length = 0
            running_reward = 0
            running_sr = 0
        episode_reward = 0

        if i_episode > args.num_episodes:
            break


def wurl(args, env, writer, **kwargs):
    obs_shape_list = env.observation_space.shape
    obs_shape = reduce((lambda x,y: x*y), obs_shape_list)
    action_space = env.action_space
    discrete_action = hasattr(action_space, 'n')
    
    if args.algo == "sac":
        if discrete_action:
            from algo.sacd import SACDTrainer
            trainers = [SACDTrainer(obs_shape, action_space, args) for _ in range(args.num_modes)]
        else:
            from algo.sac import SACTrainer
            trainers = [SACTrainer(obs_shape, action_space, args) for _ in range(args.num_modes)]
    else: # PPO
        if discrete_action:
            from algo.ppo import PPOTrainer
            trainers = [PPOTrainer(obs_shape, action_space.n, args) for _ in range(args.num_modes)]
        else:
            from algo.ppoc import PPOCTrainer
            trainers = [PPOCTrainer(obs_shape, action_space.shape[0], args) for _ in range(args.num_modes)]
        args.target_update_interval = 1000
        args.updates_per_step = 5

    memories = [ReplayMemory(args.buffer_limit) for _ in range(args.num_modes)]
    caches = [Cache(args.buffer_limit) for _ in range(args.num_modes)]

    args.amortized = False
    if args.sr_algo not in ["apwd", "pwil", "pwd"]:
        raise ValueError("Only primal WDE methods are supported!")
    elif args.sr_algo == "apwd":
        args.amortized = True
        from algo.wsre import wsre
    elif args.sr_algo == "pwil":
        from algo.wde import pwil
        wde = pwil()
    elif args.sr_algo == "pwd":
        from algo.wde import apwd
        wde = apwd()


    model_prefix = "models/{}/{}/{}/run{}/".format(args.scenario, args.exp_name, args.sr_algo, args.run)
    exp_utils.assert_path(model_prefix)
    
    src = args.reward_scale
    rc = 0
    print("Reward scale for pseudo reward: {}".format(src))
    print("Reward scale for external reward: {}".format(rc))
    running_reward = 0
    running_sr = 0
    max_mean_sr = 0
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
        while True:
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
                if timestep % args.target_update_interval == 0:
                    for _ in range(args.updates_per_step):
                        if args.algo == "sac":
                            c1_loss, c2_loss, p_loss, ent_loss, alpha = trainers[label].update_parameters(memories[label].sample(batch_size=args.batch_size), updates)
                            writer.add_scalar('loss/critic_1', c1_loss, updates)
                            writer.add_scalar('loss/critic_2', c2_loss, updates)
                            writer.add_scalar('loss/policy', p_loss, updates)
                            writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                            writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                        else:
                            lp, lq, le = trainers[label].update_parameters(memories[label].dump(2000))
                            writer.add_scalar('loss/critic', lq, updates)
                            writer.add_scalar('loss/policy', lp, updates)
                            writer.add_scalar('loss/entropy_loss', le, updates)
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
                if timestep > args.start_steps:
                    state_batch = list(caches[label].dump(episode_steps))[0]
                    srs_list = []
                    sum_srs = []
                    for i in range(args.num_modes):
                        if i!=label:
                            try:
                                target_state_batch = list(caches[i].dump(bc))[0]
                            except:
                                # print(len(list(caches[i].dump(bc))))
                                raise ValueError(f"The length of the caches does not satisfy the minimal batch")
                            if args.amortized:
                                sr = wsre(state_batch, target_state_batch)
                                sum_srs.append(np.sum(sr))
                                srs_list.append(sr)
                            else:
                                sr = wde.estimate(state_batch, target_state_batch)
                                srs_list.append(sr)
                    if args.amortized:
                        min_dist_idx = np.argmin(sum_srs) # find the nearest policy
                        srs = srs_list[min_dist_idx]
                    else:
                        srs[-1] = min(srs_list)

                for i in range(episode_steps):
                    record = trajectory[i]
                    if args.reward_episode_scale:
                        record[3] = record[3]* rc + srs[i] * src * (bc / episode_steps) # Add pseudo reward to plain reward
                    else:
                        record[3] = record[3]* rc + srs[i] * src # encourage the agent to run longer episodes
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
        writer.add_scalar('stats/episode_reward', episode_reward, i_episode)
        writer.add_scalar('stats/episode_sr', episode_sr, i_episode)
        writer.add_scalar('stats/episode_length', episode_steps, i_episode)

        if i_episode % args.log_interval == 0:
            print("Episode: {}, length: {}, reward: {}, sr: {}".format(i_episode, int(avg_length/args.log_interval),
                int(running_reward/args.log_interval), int(running_sr/args.log_interval)))
            if running_sr/args.log_interval > max_mean_sr:
                max_mean_sr = running_sr/args.log_interval
                # periodically save models
                for x in range(len(trainers)):
                    save_trainer(trainers[x], args, model_prefix, x)
                print("Models have been saved to {}".format(model_prefix))

            avg_length = 0
            running_reward = 0
            running_sr = 0
        episode_reward = 0

        if i_episode > args.num_episodes:
            break

    
def save_trainer(trainer, args, prefix, x=None):
    if args.exp_name == "wurl":
        trainer.save_model(args.scenario, prefix=prefix, suffix="{}_{}_{}".format(args.exp_name, args.sr_algo, x), silent=True)
    else:
        trainer.save_model(args.scenario, prefix=prefix, suffix="{}_{}".format(args.exp_name, x), silent=True)
    
    
if __name__ == '__main__':
    print("------------------------------------------------------------------------------------")
    start_time, exp_seed = exp_utils.get_start_time()

    # load args
    customed_args = vars(parser_train().parse_args())

    if customed_args["scenario"] in ["HalfCheetah-v3", "AntCustom-v0", "Humanoid-v3"]:
        mujoco_env = True
        with open("mujoco_configs/train/{}.json".format(customed_args["scenario"])) as f:
            json_config = json.load(f)
    else:
        with open("common.json") as f:
            json_config = json.load(f)
            
    args_dict = {}
    args_dict.update(json_config["common"])
    if mujoco_env:
        args_dict.update(json_config["env"])
    args_dict.update(json_config[customed_args["algo"]])
    args_dict.update(json_config[customed_args["exp_name"]])
    args_dict.update(customed_args)
    args = Namespace(**args_dict)
    args.seed = exp_seed
    print(args)
    print("now is running: {} - run{}!".format(args.exp_name, args.run))

    # create env
    if args.scenario == "AntCustom-v0":
        from gym_mm.envs.ant_custom_env import AntCustomEnv
        env = AntCustomEnv()
        reduced_obs = True
    else:
        env = gym.make(args.scenario)
        reduced_obs = False
    
    # set up seed
    exp_utils.setup_seed(args.seed)
    env.seed(args.seed)

    # creating logging file
    if args.exp_name == "wurl":
        logdir = 'logs/{}/{}/{}/{}_run{}_{}'.format(args.scenario, args.exp_name, args.sr_algo, args.algo, args.run, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        logdir = 'logs/{}/{}/{}_run{}_{}'.format(args.scenario, args.exp_name, args.algo, args.run, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        
    writer = SummaryWriter(logdir=logdir)

    # run experiment
    eval(args.exp_name)(args, env, writer, reduced_obs=reduced_obs)
    env.close()
    
    now_time = datetime.datetime.now().strftime("%Y.%m.%d - %H:%M:%S")
    exp_time = "{} -> {}".format(start_time, now_time)
    
    exp_utils.save_args(json_config, args, logdir)
    exp_utils.save_exp_config(args, exp_time, args.exp_name)
    
    print("This experiment time: {}".format(exp_time))
    print("------------------------------------------------------------------------------------")
