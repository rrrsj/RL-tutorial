import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import copy
import random
from torch.utils.tensorboard import SummaryWriter

class Q_value(nn.Module):
    def __init__(self,input_dim,latent_dim,output_dim):
        super(Q_value,self).__init__()
        self.input_dim=input_dim
        self.latent_dim=latent_dim
        self.output_dim=output_dim
        self.now_model=nn.Sequential(nn.Linear(self.input_dim,self.latent_dim),
        nn.GELU(),
        nn.Linear(self.latent_dim,self.latent_dim),
        nn.GELU(),
        nn.Linear(self.latent_dim,self.output_dim))

    def forward(self,inputs):
        return self.now_model(inputs)

class Experience:
    def __init__(self):
        self.experience=[]
    def add_experience(self,old_state,reward,new_state,action,end):
        experience_now=[old_state,reward,new_state,action,end]
        self.experience.append(experience_now)
        if len(self.experience)>100000:
            self.experience=random.sample(self.experience, 100000)
    def get_experience(self,batch_size):
        batch_size=min(batch_size,len(self.experience))
        batch = random.sample(self.experience, batch_size)
        old_state=[batch[i][0] for i in range(len(batch))]
        reward=[batch[i][1] for i in range(len(batch))]
        new_state=[batch[i][2] for i in range(len(batch))]
        new_action=[batch[i][3] for i in range(len(batch))]
        end=[batch[i][4] for i in range(len(batch))]
        return old_state,reward,new_state,new_action,end


class DQN:
    def __init__(self,device):
        self.device=device
        self.time_gamma=0.99
        self.env=gym.make("CartPole-v1")
        self.now_step=0
        self.buffer=Experience()
        self.qvalue=Q_value(4,512,2).to(self.device)
        self.optimizer=torch.optim.AdamW(self.qvalue.parameters(),lr=1e-4)
    
    def sample_data(self):
        obs,_=self.env.reset()
        action=[0,1]
        while 1:
            old_obs=copy.deepcopy(obs)
            if random.uniform(0.,1.)<=0.7:
                now_action=torch.max(self.qvalue(torch.tensor(obs).to(self.device)),dim=-1)[1].item()
            else:
                action_prob_temp=[0.5,0.5]
                now_action=np.random.choice(action,p=action_prob_temp)
            obs,reward,terminated,truncated,_=self.env.step(now_action)
            end=0
            if terminated or truncated:
                end=1
            self.buffer.add_experience(old_obs,reward,obs,now_action,end)
            if terminated or truncated:
                break


    def DQN_Train(self):
        old_state,reward,new_state,new_action,end=self.buffer.get_experience(512)
        old_state=torch.tensor(old_state).to(self.device)
        reward=torch.tensor(reward).to(self.device).unsqueeze(1)
        new_state=torch.tensor(new_state).to(self.device)
        new_action=torch.tensor(new_action).to(self.device).unsqueeze(1)
        end=torch.tensor(end).to(self.device).unsqueeze(1)

        now_action_value=self.qvalue(old_state)
        next_action_value=self.qvalue(new_state)
        result = torch.gather(now_action_value, dim=1, index=new_action)
        next_action_value=next_action_value.max(dim=-1,keepdim=True)[0]
        next_action_value=next_action_value*(1-end)
        loss=((result-reward-self.time_gamma*next_action_value)**2).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


   
    def get_test_reward(self,writer):
        all_reward=0
        obs,_=self.env.reset()
        while 1:
            with torch.no_grad():
                now_action=torch.max(self.qvalue(torch.tensor(obs).to(self.device)),dim=-1)[1].item()
                obs,reward,terminated,truncated,info=self.env.step(now_action)
                all_reward=all_reward+reward
                if terminated or truncated:
                    break 
        writer.add_scalar('reward',all_reward,global_step=self.now_step,walltime=None)
        self.now_step=self.now_step+1

writer=SummaryWriter('./log')
model=DQN('cuda')

while 1:
    model.sample_data()
    for i in range(100):
        model.DQN_Train()
    model.get_test_reward(writer)