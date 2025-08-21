import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import copy
import random
from torch.utils.tensorboard import SummaryWriter

class Actor(nn.Module):
    def __init__(self,input_dim,latent_dim,output_dim):
        super(Actor,self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.latent_dim=latent_dim
        self.model=nn.Sequential(nn.Linear(self.input_dim,self.latent_dim),
        nn.GELU(),
        nn.Linear(self.latent_dim,self.output_dim))
    
    def forward(self,inputs):
        return torch.softmax(self.model(inputs),dim=-1)

class Critic(nn.Module):
    def __init__(self,input_dim,latent_dim,output_dim):
        super(Critic,self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.latent_dim=latent_dim
        self.model=nn.Sequential(nn.Linear(self.input_dim,self.latent_dim),
        nn.GELU(),
        nn.Linear(self.latent_dim,self.output_dim))
    
    def forward(self,inputs):
        return self.model(inputs)


class Experience:
    def __init__(self):
        self.buffer=[]
    def add_experience(self,experience_now):
        self.buffer.append(experience_now)
    def get_experience(self,batch_size):
        batch_size=min(batch_size,len(self.buffer))
        batch = random.sample(self.buffer, batch_size)
        action_p=[batch[i][0].tolist() for i in range(len(batch))]
        A=[batch[i][1].detach().to('cpu').tolist() for i in range(len(batch))]
        obs=[batch[i][2] for i in range(len(batch))]
        action=[batch[i][3] for i in range(len(batch))]
        critic_target=[batch[i][4].detach().to('cpu').tolist() for i in range(len(batch))]
        return action_p,A,obs,action,critic_target



class PPO:
    def __init__(self,device):
        self.device=device
        self.gamma=0.99
        self.env=gym.make("CartPole-v1")
        self.now_step=0
        self.epslion=0.2
        self.buffer=Experience()
        self.actor=Actor(4,64,2).to(self.device)
        self.critic=Critic(4,64,1).to(self.device)
        self.optimizer_a=torch.optim.AdamW(self.actor.parameters(),lr=1e-3)
        self.optimizer_c=torch.optim.AdamW(self.critic.parameters(),lr=1e-2)

    def sample_data(self):
        obs,_=self.env.reset()
        action=[0,1]

        with torch.no_grad():
            while 1:
                old_obs=copy.deepcopy(obs)
                experience_now=[]
                action_prob=self.actor(torch.tensor(obs).to(self.device)).to('cpu').numpy()
                action_prob[-1]=1.-action_prob[:-1].sum()
                now_action=np.random.choice(action,p=action_prob)
                obs,reward,terminated,truncated,info=self.env.step(now_action)
                next_value=self.critic(torch.tensor(obs).to(self.device))
                if terminated or truncated:
                    next_value=torch.tensor([0.]).to(self.device)
                experience_now=[action_prob,reward+self.gamma*next_value-self.critic(torch.tensor(old_obs).to(self.device)),obs,now_action,reward+self.gamma*next_value]
                self.buffer.add_experience(experience_now)
                if terminated or truncated:
                    break

    def PPO_train(self):
        old_p,A,obs,action,critic_target=self.buffer.get_experience(512)
        batch=len(old_p)
        old_p=torch.tensor(old_p).to(self.device).reshape(batch,-1)
        A=torch.tensor(A).to(self.device).reshape(batch,-1)
        obs=torch.tensor(obs).to(self.device).reshape(batch,-1)
        action=torch.tensor(action).to(self.device).reshape(batch,-1)
        critic_target=torch.tensor(critic_target).to(self.device).reshape(batch,-1)

        new_p=self.actor(obs)
        old_p=torch.gather(old_p, dim=1,index=action)
        new_p=torch.gather(new_p, dim=1,index=action)
        resample=new_p/old_p
        surrogate = resample * A
        clipped = torch.clamp(resample,1-self.epslion,1+self.epslion) * A

        loss_a=-(torch.min(resample*A,clipped*A)).mean()

        loss_c=(resample.detach()*(self.critic(obs)-critic_target)**2).mean()
        loss_a.backward()
        loss_c.backward()
        self.optimizer_a.step()
        self.optimizer_c.step()
        self.optimizer_a.zero_grad()
        self.optimizer_c.zero_grad()
        

   
    def get_test_reward(self,writer):
        all_reward=0
        obs,_=self.env.reset()
        while 1:
            with torch.no_grad():
                now_action=torch.max(self.actor(torch.tensor(obs).to(self.device)),dim=-1)[1].item()
                obs,reward,terminated,truncated,info=self.env.step(now_action)
                all_reward=all_reward+reward
                if terminated or truncated:
                    break 
        writer.add_scalar('reward',all_reward,global_step=self.now_step,walltime=None)
        self.now_step=self.now_step+1

writer=SummaryWriter('./log')
model=PPO('cuda')

while 1:
    model.buffer.buffer=[]
    for i in range(10):
        model.sample_data()
    for i in range(10):
        model.PPO_train()
    model.get_test_reward(writer)