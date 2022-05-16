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
from tensorboardX import SummaryWriter
from gym_mm.envs.freerun import FreeRun
from gym_mm.envs.freerund import FreeRunD
from mujoco_maze import PointEnv
from algo.buffer import ReplayMemory, Cache
from config import parser_main
from argparse import Namespace

start_time = datetime.datetime.now()

customed_args = vars(parser_main().parse_args())
with open("mujoco_configs/{}.json".format(customed_args['scenario'])) as f:
    json_config = json.load(f)

args_dict = {}
args_dict.update(customed_args)
args_dict.update(json_config["common"])
args_dict.update(json_config[args_dict["sr_algo"]])

# reupdate "reward_scale" from customed args
args_dict.update({"reward_scale": customed_args["reward_scale"]
                  })

args = Namespace(**args_dict)
print(args)
print("now is run: {}!".format(args.run))

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Load environment
if args.scenario == "Ant-v3":
    if args.ant_custom:
        # add ant_custom
        # from envs.gym_env import GymEnv
        from envs.ant_custom_env import AntCustomEnv
        env = AntCustomEnv(args.ant_custom_gear_ratio)
        # env = GymEnv(env=env)
        # if variant.pop('normalize_env'):  # without normalize process
        #     env = normalize(env)
    else:
        # add xy position in observation for Ant-v3
        gym.make(args.scenario, exclude_current_positions_from_observation=args.exclude_current_positions_from_observation)
else:
    env = gym.make(args.scenario)

# env = gym.make(args.scenario) if args.scenario != "Ant-v3" else gym.make(
#     args.scenario,
#     exclude_current_positions_from_observation=args.
#     exclude_current_positions_from_observation
# )  # add xy position in observation for Ant-v3

env.seed(args.seed)
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
if not os.path.exists("logs/{}".format(args.scenario)):
    os.makedirs("logs/{}".format(args.scenario))

logdir = 'logs/{}/wasserstein_{}_{}_run{}_{}'.format(args.scenario, args.algo, args.scenario, args.run, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
writer = SummaryWriter(logdir=logdir)

run = args.run
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
            if timestep % update_interval == 0:
                for _ in range(updates_per_step):
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
                            print(len(list(caches[i].dump(bc))))
                        if amortized:
                            # Use x,y position to calculate the sr reward while the scenario is "Ant-v3"
                            if args.scenario in ["Ant-v3"] and args.use_xy_for_sr_reward:
                                sr = wsre(state_batch[:, :2], target_state_batch[:, :2])
                            else:
                                sr = wsre(state_batch, target_state_batch)
                            sum_srs.append(np.sum(sr))
                            srs_list.append(sr)
                        else:
                            sr = wde.estimate(state_batch, target_state_batch)
                            srs_list.append(sr)
                if amortized:
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
                trainers[x].save_model(args.scenario, prefix="models/{}/wasserstein_{}/run{}/".format(args.scenario, args.sr_algo, run), suffix="wasserstein_{}_{}".format(args.sr_algo, x), silent=True)
            print("Models saved to "+"models/{}/wasserstein_{}/run{}/".format(args.scenario, args.sr_algo, run))

        avg_length = 0
        running_reward = 0
        running_sr = 0
    episode_reward = 0

    if i_episode > args.num_episodes:
        break

for x in range(len(trainers)):
    trainers[x].save_model(args.scenario, prefix="models/{}/wasserstein_{}/run{}/".format(args.scenario, args.sr_algo, run), suffix="wasserstein_{}_{}".format(args.sr_algo, x))
print("Models saved to "+"models/{}/wasserstein_{}/run{}/".format(args.scenario, args.sr_algo, run))
env.close()

now_time = datetime.datetime.now()
print("This experiment time: {} -> {}".format(start_time.strftime("%Y.%m.%d - %H:%M:%S"),
                                              now_time.strftime("%Y.%m.%d - %H:%M:%S")))
