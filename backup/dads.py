import argparse
import torch
import numpy as np 
import itertools
import datetime
import gym
from functools import reduce
from copy import deepcopy
from gym_mm.envs.freerun import FreeRun
from algo.buffer import ReplayMemory
from algo.disc import PredTrainer
from algo.utils import convert_to_onehot, wrapped_obs
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description="DADS")
parser.add_argument('--scenario', type=str, default="AntCustom-v0", help="environment")
parser.add_argument('--num_episodes', type=int, default=2000, help="number of episodes for training")
parser.add_argument('--max_episode_len', type=int, default=100, help="maximum episode length")
parser.add_argument('--hidden_dim', type=int, default=64, help="network hidden size")
parser.add_argument('--log_interval', type=int, default=20, help="calculate avg reward every log_interval episodes")
parser.add_argument('--updates_per_step', type=int, default=1, help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--seed', type=int, default=123, help="random seed for env")
parser.add_argument('--buffer_limit', type=int, default=100000, help="an imagined limit for replay buffer")
parser.add_argument('--tau', type=float, default=0.005, help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--beta1', type=float, default=0.9, help="beta1 for adam optimizer")
parser.add_argument('--beta2', type=float, default=0.999, help="beta2 for adam optimizer")
parser.add_argument('--beta', type=float, default=0., help="beta for entropy term")
parser.add_argument('--gamma', type=float, default=0.999, help="discounted factor")
parser.add_argument('--critic_lr', type=float, default=0.0003, help="lr for the critic")
parser.add_argument('--policy_lr', type=float, default=0.0003, help="lr for the policy")  
parser.add_argument('--alpha', type=float, default=0.2, help='Temperature parameter α determines the relative importance of the entropy\
                        term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, help='Automaically adjust α (default: False)')
parser.add_argument('--batch_size', type=int, default=200, help="maximum number of steps retrieved from buffer")
parser.add_argument('--disc_batch_size', type=int, default=200, help="batch size for predictor")
parser.add_argument('--pred_lr', type=float, default=0.00003, help="learning rate for predictor")
parser.add_argument('--cuda', action='store_false', help='run on GPU (default: True)')
parser.add_argument('--policy', type=str, default="Gaussian", help="policy type for backend")
parser.add_argument('--num_modes', type=int, default=10, help="number of modes that need to learn")
parser.add_argument('--reward_scale', type=float, default=100, help="scale factor for pseudo reward")
parser.add_argument('--algo', type=str, default='sac', help="training algorithm for agent learning")
parser.add_argument('--schedule', type=str, default='random', help="learning schedule for policy modes")
parser.add_argument('--run', type=int, default=10, help="index of individual runs")

args = parser.parse_args()
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
trainer = SACTrainer(obs_shape + args.num_modes, action_space, args)
update_interval = 1
updates_per_step = 1
memory = ReplayMemory(args.buffer_limit)

# TensorboardX
logdir = 'logs_new/dads_{}_{}_{}'.format(args.algo, args.scenario, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
writer = SummaryWriter(logdir=logdir)

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
input_amp = 1000
for i_episode in itertools.count(1):
    obs = env.reset()
    episode_reward = 0
    episode_sr = 0
    episode_steps = 0
    label = np.random.randint(0, high=args.num_modes)
    l = np.array([label])
    for t in range(args.max_episode_len):
        wobs = wrapped_obs(obs, l, args.num_modes)
        if args.start_steps < timestep:
            action, logprob = trainer.act(wobs)
            if len(action.shape) > 1:
                action = action[0]
            if discrete_action:
                action = action[0]
        else:
            action = env.action_space.sample()
            logprob = np.array([1.0])

        if len(memory) > args.start_steps:
            if timestep % update_interval == 0:
                for _ in range(args.updates_per_step):
                    label_batch, state_batch, action_batch, logprob_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=args.batch_size)
                    wstate_batch = wrapped_obs(state_batch, label_batch, args.num_modes)
                    wnext_state_batch = wrapped_obs(next_state_batch, label_batch, args.num_modes)

                    c1_loss, c2_loss, p_loss, ent_loss, alpha = trainer.update_parameters((wstate_batch, action_batch, logprob_batch, reward_batch, wnext_state_batch, mask_batch), updates)
                    writer.add_scalar('loss/critic_1', c1_loss, updates)
                    writer.add_scalar('loss/critic_2', c2_loss, updates)
                    writer.add_scalar('loss/policy', p_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)                        
                 
                    updates += 1
            label_batch, state_batch, _, _, _, next_state_batch, _ = memory.sample(batch_size=args.disc_batch_size)
            label_onehot_batch = convert_to_onehot(label_batch, args.num_modes)
            state_delta_batch = next_state_batch - state_batch
            if reduced_obs:
                d_loss = dtrainer.update_parameters((state_batch[:,:2], label_onehot_batch, state_delta_batch[:,:2] * input_amp))
            else:
                d_loss = dtrainer.update_parameters((state_batch, label_onehot_batch, state_delta_batch * input_amp))
            writer.add_scalar('loss/disc', d_loss, timestep)

        new_obs, reward, done, _ = env.step(action.tolist())

        if len(memory) > args.start_steps:
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

        memory.push((label, obs, action, logprob, sr * src + reward * rc, new_obs, mask))
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
            trainer.save_model(args.scenario, prefix="models/{}/run{}/".format(args.scenario, args.run), suffix="dads", silent=True)
        avg_length = 0
        running_reward = 0
        running_sr = 0
    episode_reward = 0

    if i_episode > args.num_episodes:
        break

# trainer.save_model(args.scenario, suffix="dads")
print("Models saved to "+"models/{}/run{}/".format(args.scenario, args.run))
env.close()
