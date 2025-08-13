import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

from torch.utils.tensorboard import SummaryWriter

class policy_network(nn.Module):
    def __init__(self,input_dim,latent_dim,output_dim):
        super(policy_network,self).__init__()
        self.input_dim=input_dim
        self.latent_dim=latent_dim
        self.output_dim=output_dim
        self.now_model=nn.Sequential(nn.Linear(self.input_dim,self.latent_dim),
        nn.GELU(),
        nn.Linear(self.latent_dim,self.output_dim))

    def forward(self,inputs):
        return torch.softmax(self.now_model(inputs),dim=-1)

class S_value(nn.Module):
    def __init__(self,input_dim,latent_dim,output_dim):
        super(S_value,self).__init__()
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
        self.policy=policy_network(4,64,2).to(self.device)
        self.S_value=S_value(4,64,1).to(self.device)
        self.optimizer_p=torch.optim.AdamW(self.policy.parameters(),lr=1e-4)
        self.optimizer_s=torch.optim.AdamW(self.S_value.parameters(),lr=1e-4)

    def get_train_experience(self):
        self.experience=[]
        obs,_=self.env.reset()
        action=[0,1]
        while 1:
            with torch.no_grad():
                step_experience=[obs]
                action_prob=self.policy(torch.tensor(obs).to(self.device)).to('cpu').numpy()
                action_prob[-1]=1.-action_prob[:-1].sum()
                now_action=np.random.choice(action,p=action_prob)
                obs,reward,terminated,truncated,info=self.env.step(now_action)
                step_experience.append(now_action)
                step_experience.append(reward)
                self.experience.append(step_experience)
                if terminated or truncated:
                    break

    def get_true_reward(self):
        for i in range(len(self.experience)-1):
            self.experience[len(self.experience)-2-i][2]=self.time_gamma*self.experience[len(self.experience)-1-i][2]+self.experience[len(self.experience)-2-i][2]
    
    def train_s_value(self,writer):
        #Monte Carlo sampling itself has high variance and can be modified into the TD algorithm
        for i in range(len(self.experience)):
            self.optimizer_s.zero_grad()
            s_value=self.S_value(torch.tensor(self.experience[i][0]).to(self.device))
            loss=((s_value-self.experience[i][2])**2).sum()
            loss.backward()
            self.optimizer_s.step()
            writer.add_scalar('s_loss',loss.item(),global_step=self.now_step_s,walltime=None)
            self.now_step_s=self.now_step_s+1

    def train_policy(self):
        for i in range(len(self.experience)):
            self.optimizer_p.zero_grad()
            action_prob=self.policy(torch.tensor(self.experience[i][0]).to(self.device))
            with torch.no_grad():
                baseline_value=self.S_value(torch.tensor(self.experience[i][0]).to(self.device))
            loss=-(self.experience[i][2]-baseline_value.item())*torch.log(action_prob[self.experience[i][1]])
            loss.backward()
            self.optimizer_p.step()
            
    def get_test_reward(self,writer):
        all_reward=0
        obs,_=self.env.reset()
        while 1:
            with torch.no_grad():
                now_action=torch.max(self.policy(torch.tensor(obs).to(self.device)),dim=-1)[1].item()
                obs,reward,terminated,truncated,info=self.env.step(now_action)
                all_reward=all_reward+reward
                if terminated or truncated:
                    break 

        writer.add_scalar('reward',all_reward,global_step=self.now_step,walltime=None)
        self.now_step=self.now_step+1

writer=SummaryWriter('./log')
model=PL_GRA('cuda')

for i in range(1000):
    model.get_train_experience()
    model.get_true_reward()
    model.train_s_value(writer)

while 1:
    model.get_train_experience()
    model.get_true_reward()
    model.train_policy()
    model.get_test_reward(writer)
