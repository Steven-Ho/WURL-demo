import os
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import numpy as np
from .model import GaussianPolicy, VNetwork
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

# class ActorCriticB(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim, action_std, args):
#         super(ActorCriticB, self).__init__()
#         self.actor = GaussianPolicy(state_dim, action_dim, hidden_dim, action_std=action_std)
#         self.critic = VNetwork(state_dim, hidden_dim)
#         self.device = torch.device("cuda" if args.cuda else "cpu")

#     def forward(self):
#         raise NotImplementedError
    
#     def act(self, state):
#         action, action_logprob, _ = self.actor.sample(state)
        
#         return action.detach(), action_logprob.detach()
    
#     def evaluate(self, state, action):   
#         action_logprobs, dist_entropy = self.actor.evaluate(state, action)
#         state_value = self.critic(state)
        
#         return action_logprobs, torch.squeeze(state_value), dist_entropy

#     def to(self, device):
#         self.actor = self.actor.to(device)
#         self.critic = self.critic.to(device)
#         return super(ActorCriticB, self).to(device)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, args, action_std=None):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.action_std = action_std
        self.mean =  nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh()
                )
        if action_std is None:
            self.log_std = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, action_dim)
                    ) 
                     
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
                )
        self.device = torch.device("cuda" if args.cuda else "cpu")
        if action_std is not None:
            self.action_var = torch.full((action_dim,), action_std*action_std).to(self.device)
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        action_mean = self.mean(state)
        if self.action_std is None:           
            action_log_std = self.log_std(state)
            action_log_std = torch.clamp(action_log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            action_var = action_log_std.exp().pow(2).squeeze()
            cov_mat = torch.diag(action_var).to(self.device)
        else:
            cov_mat = torch.diag(self.action_var).to(self.device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):   
        action_mean = self.mean(state)
        
        if self.action_std is None:
            action_log_std = self.log_std(state)
            action_log_std = torch.clamp(action_log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            action_var = action_log_std.exp().pow(2)
        else:
            action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPOCTrainer:
    def __init__(self, obs_shape, action_shape, args):
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.gamma = args.gamma
        self.eps_clip = args.eps_clip
        self.K_epochs = args.K_epochs

        self.device = torch.device("cuda" if args.cuda else "cpu")
        
        self.policy = ActorCritic(obs_shape, action_shape, args.hidden_dim, args, args.action_std).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        
        self.policy_old = ActorCritic(obs_shape, action_shape, args.hidden_dim, args, args.action_std).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def act(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action, action_logprobs = self.policy_old.act(state)
        return action.cpu().data.numpy().flatten(), action_logprobs.cpu().data.numpy().flatten()
    
    def update_parameters(self, samples):
        # Unpack batch
        obs, action, old_log_p, reward, _, done = samples

        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(list(reward)), reversed(list(done))):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.FloatTensor(obs).to(self.device)
        old_actions = torch.FloatTensor(action).to(self.device)
        old_logprobs = torch.FloatTensor(old_log_p).squeeze().to(self.device)
        
        # Optimize policy for K epochs:
        loss_p = []
        loss_q = []
        loss_e = []
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            lp = -torch.min(surr1, surr2).mean()
            lq = 0.5*self.MseLoss(state_values, rewards)
            le = -0.01*dist_entropy.mean()
            loss_p.append(lp.item())
            loss_q.append(lq.item())
            loss_e.append(le.item())
            loss = lp + lq + le
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        return sum(loss_p)/self.K_epochs, sum(loss_q)/self.K_epochs, sum(loss_e)/self.K_epochs

    def save_model(self, env_name, prefix="models/", suffix="", path=None, silent=False):
        if not os.path.exists(prefix):
            os.makedirs(prefix)

        if path is None:
            path = prefix+"ppo_{}_{}".format(env_name, suffix)
        if not silent:
            print('Saving models to {}'.format(path))
        torch.save(self.policy_old.state_dict(), path)

    def load_model(self, path):
        print('Loading models from {}'.format(path))
        if path is not None:
            self.policy_old.load_state_dict(torch.load(path))