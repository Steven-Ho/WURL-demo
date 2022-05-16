import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from .utils import soft_update, hard_update
from .model import GaussianPolicy, QNetwork, CategoricalPolicy

EPS = 1e-6
class SACDTrainer(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        action_shape = action_space.n

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu") 

        self.critic = QNetwork(num_inputs, action_shape, args.hidden_dim).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.critic_lr)

        self.critic_target = QNetwork(num_inputs, action_shape, args.hidden_dim).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.policy = CategoricalPolicy(num_inputs, action_shape, args.hidden_dim).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.policy_lr)

        self.automatic_entropy_tuning = False

    def act(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, log_prob, _ = self.policy.sample(state)
        else:
            _, log_prob, action = self.policy.sample(state)
        return action.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

    def update_parameters(self, samples, updates):
        # Sample a batch from memory
        state_batch, action_batch, _, reward_batch, next_state_batch, mask_batch = samples

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            _, next_state_pi = self.policy.forward(next_state_batch)
            next_state_log_pi = torch.log(next_state_pi + EPS)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            B, D = next_state_pi.shape
            v_next_state = torch.bmm(next_state_pi.view(B, 1, D), (min_qf_next_target - self.alpha * next_state_log_pi).view(B, D, 1))
            next_q_value = reward_batch + mask_batch * self.gamma * v_next_state.squeeze(dim=-1)
            action_batch = F.one_hot(action_batch.long(), num_classes=D).float()

        qf1, qf2 = self.critic(state_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_pi = torch.bmm(action_batch.view(B, 1, D), qf1.view(B, D, 1))
        qf2_pi = torch.bmm(action_batch.view(B, 1, D), qf2.view(B, D, 1))
        qf1_loss = F.mse_loss(qf1_pi.squeeze(), next_q_value.squeeze()) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2_pi.squeeze(), next_q_value.squeeze()) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        _, next_state_pi = self.policy.forward(state_batch)
        state_log_pi = torch.log(next_state_pi + EPS)
        min_qf = torch.min(qf1, qf2).detach()
        policy_loss = torch.bmm(action_batch.view(B, 1, D), (self.alpha * state_log_pi - min_qf).view(B, D, 1)).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        alpha_loss = torch.tensor(0.).to(self.device)
        alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, env_name, prefix="models/", suffix="", actor_path=None, critic_path=None, silent=False):
        if not os.path.exists(prefix):
            os.makedirs(prefix)

        if actor_path is None:
            actor_path = prefix+"sacd_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = prefix+"sacd_critic_{}_{}".format(env_name, suffix)
        if not silent:
            print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
    
    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

