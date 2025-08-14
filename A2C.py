import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

from torch.utils.tensorboard import SummaryWriter

class action(nn.Module):
    def __init__(self,input_dim,latent_dim,output_dim):
        super(action,self).__init__()
        self.input_dim=input_dim
        self.latent_dim=latent_dim
        self.output_dim=output_dim
        self.now_model=nn.Sequential(nn.Linear(self.input_dim,self.latent_dim),
        nn.GELU(),
        nn.Linear(self.latent_dim,self.output_dim))

    def forward(self,inputs):
        return torch.softmax(self.now_model(inputs),dim=-1)

class critic(nn.Module):
    def __init__(self,input_dim,latent_dim,output_dim):
        super(critic,self).__init__()
        self.input_dim=input_dim
        self.latent_dim=latent_dim
        self.output_dim=output_dim
        self.model=nn.Sequential(nn.Linear(self.input_dim,self.latent_dim),
        nn.GELU(),
        nn.Linear(self.latent_dim,self.output_dim))
    def forward(self,inputs):
        return self.model(inputs)

class PL_GRA:
    def __init__(self,device):
        self.device=device
        self.time_gamma=0.99
        self.now_step=0
        self.now_step_s=0
        self.env=gym.make("CartPole-v1")
        self.action=action(4,64,2).to(self.device)
        self.critic=critic(4,64,1).to(self.device)
        self.optimizer_action=torch.optim.AdamW(self.action.parameters(),lr=1e-3)
        self.optimizer_critic=torch.optim.AdamW(self.critic.parameters(),lr=1e-3)

    def TD_train(self):
        obs,_=self.env.reset()
        action_type=[0,1]
        while 1:
            dw=0
            action_prob=self.action(torch.tensor(obs).to(self.device))
            now_state_value=self.critic(torch.tensor(obs).to(self.device))
            action_prob_temp=action_prob.to('cpu').detach().numpy()
            action_prob_temp[-1]=1-action_prob_temp[:-1].sum()
            now_action=np.random.choice(action_type,p=action_prob_temp)
            obs,reward,terminated,truncated,info=self.env.step(now_action)
            if terminated or truncated:
                dw=1
            
            next_state_value=self.critic(torch.tensor(obs).to(self.device))

            value_loss=((now_state_value-reward-(1-dw)*self.time_gamma*next_state_value)**2).sum()
            action_loss=(-(reward+self.time_gamma*next_state_value.item()-now_state_value.item())*torch.log(action_prob[now_action])).sum()
            value_loss.backward()
            action_loss.backward()
            self.optimizer_action.step()
            self.optimizer_critic.step()
            self.optimizer_action.zero_grad()
            self.optimizer_critic.zero_grad()
            if terminated or truncated:
                break
            
            
    def get_test_reward(self,writer):
        all_reward=0
        obs,_=self.env.reset()
        while 1:
            with torch.no_grad():
                now_action=torch.max(self.action(torch.tensor(obs).to(self.device)),dim=-1)[1].item()
                obs,reward,terminated,truncated,info=self.env.step(now_action)
                all_reward=all_reward+reward
                if terminated or truncated:
                    break 

        writer.add_scalar('reward',all_reward,global_step=self.now_step,walltime=None)
        self.now_step=self.now_step+1

writer=SummaryWriter('./log')
model=PL_GRA('cuda')

while 1:
    model.TD_train()
    model.get_test_reward(writer)
