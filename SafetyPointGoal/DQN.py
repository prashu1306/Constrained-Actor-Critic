# %matplotlib inline
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.distributions import Categorical
import numpy as np
from numpy.random import randint
from numpy.random import choice
from matplotlib import pyplot as plt
import gym
import safety_gymnasium as sg
import random as rnd
from gym.wrappers import FlattenObservation
import math
from collections import deque
import statistics
class valuefunction(nn.Module):
    def __init__(self,n1,n2,n3):
        super(valuefunction,self).__init__()
        self.l1 = nn.Linear(n1,n2)
        self.l2 = nn.Linear(n2,n3)
        
    def forward(self,x):
        x =self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        return x
    
class policyparameter(nn.Module):
    def __init__(self,n1,n2,n3):
        super(policyparameter,self).__init__()
        self.l1 = nn.Linear(n1,n2)
        self.l2 = nn.Linear(n2,n3)
        
    def forward(self,x):
         x =self.l1(x)
         x = torch.relu(x)
         x = self.l2(x)
         x = torch.softmax(x, dim=0)
         return x
    
class QNN(nn.Module):
    def __init__(self,n1,n2,n3):
        super(QNN,self).__init__()
        self.l1 = nn.Linear(n1,n2)
        self.l2 = nn.Linear(n2,n3)
        self.lossfn = nn.MSELoss()

    def forward(self,x):
        x =self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        return x
    
 
env = sg.make("SafetyPointGoal1-v0")




d1 = 60
d2 = 10
d3 =  1
nA = 4
n_train = 5000
N = 5000


    
def feat(state):
    res = torch.zeros(d1)
    for i in range(d1):
        res[i]=torch.as_tensor(state[i])
    return res
        


  
def getaction(action):
    a = 0
    b = 0
    c = 0
    if(action == 0):
        a = rnd.uniform(-1,-0.5)
        b = rnd.uniform(-1,-0.5)
        c = rnd.uniform(-1,-0.5)
    if(action ==1):
        a = rnd.uniform(-0.5,0)
        b = rnd.uniform(-0.5,0)
        c = rnd.uniform(-0.5,0)
    if(action ==2 ):
        a = rnd.uniform(0,0.5)
        b = rnd.uniform(0,0.5)
        c = rnd.uniform(0,0.5)
    if(action == 3):
        a = rnd.uniform(0.5,1)
        b = rnd.uniform(0.5,1)
        c = rnd.uniform(0.5,1)
    
    return torch.tensor([a,b])

        


  


def train_dqn():
   Qvalue = QNN(d1,d2,nA)
   qoptim = torch.optim.AdamW(Qvalue.parameters(),lr = 0.5,amsgrad=True)
   state , info= env.reset()
   J = 0
   const = 0
  

   n = 1

   L = 0
   epsilon = 0.04
   gamma = 0
   alpha = 0.1
   P = 10000000
   Y =0
   Qvaluee = Qvalue(feat(state)).detach().numpy()
   while n <= n_train:
       if np.random.uniform(0,1) < epsilon:
           action = np.random.choice(range(nA))
       else:
           action = np.argmax(Qvaluee)
       action_ = getaction(action)
       next_state,reward,cost ,terminated,truncated ,info= env.step(action_)
       if(terminated or truncated):
            next_state,info = env.reset()
            
       a = 1.5/(n**0.4)
       c = 1.5/(n**0.8)     
       J = (J + reward)/n #average reward
       L += a*(reward- gamma*(cost - alpha)-L)
       Qvalue_next = Qvalue(feat(next_state))
       qv = torch.max(Qvalue_next)
       Qvaluee_next = torch.tensor([qv])
       reward_tensor = torch.tensor([reward - gamma*(cost - alpha)])
       L_tensor = torch.tensor([L])
       criterion = nn.SmoothL1Loss()
       Y = Y + a*(cost - Y)
       gamma  = max(0, min(P, gamma + c * (Y - alpha)))
       state_action_values = Qvalue(feat(state))[action]
       expected_state_action_values = reward_tensor - L_tensor + Qvaluee_next
       qloss = criterion(state_action_values,expected_state_action_values)
       #print(qloss.mean().shape)
       qoptim.zero_grad()
       qloss.backward()
       qoptim.step()
       n += 1
       state = next_state
   return Qvaluee  

   

def DQN(seed):
   np.random.seed(seed)
   Qvaluee = train_dqn()
   indices = []
   reward_list =[]
   const_list =[]
   returns = deque(maxlen = N)
   returns_c = deque(maxlen = N)
   epsilon = 0.4
   m = 1
   while m <= N:
       if np.random.uniform(0,1) < epsilon:
           action = np.random.choice(range(nA))
       else:
           action = np.argmax(Qvaluee)
       action_ = getaction(action)
       next_state,reward,cost ,terminated,truncated ,info= env.step(action_)
       if(terminated or truncated):
            #print('obstacle reached' , n)
            next_state,info = env.reset()
       returns.append(reward)
       returns_c.append(cost)
       reward_list.append(np.mean(returns))
       const_list.append(np.mean(returns_c))
       #print(J)
       indices.append(m)
       
       m += 1
       state = next_state
   return reward_list,const_list,indices


  
f = open("plotting_dqn_final.txt", "a")
f1 = open("plotting_dqn_sdt_final.txt", "a") 
f2 = open("plotting_dqn_const_final.txt", "a")
f3 = open("plotting_dqn_const_std_final.txt", "a")

n_seed = 10
seed = randint(1000,size = (n_seed,1))
for i in range(0,n_seed):
      seed[i] = randint(1000)

reward_list_q = np.zeros((n_seed,N)) 
const_list_q = np.zeros((n_seed,N))
    
for i in range(0,n_seed):
      reward_list_q[i] ,const_list_q[i], indices = DQN(seed[i][0])
      #reward_list_ac[i], indices = actor_critic(seed[i][0])
      #reward_list_ca[i],indices = critic_actor(seed[i][0])
      #indices , reward_list_ppo_ac[i] = PPO_actor_critic(seed[i][0])
      #indices , reward_list_ppo_ca[i] = PPO_critic_actor(seed[i][0])
      #print(i)
#reward_ac = np.mean(reward_list_ac,axis = 0)
#reward_ca = np.mean(reward_list_ca,axis = 0)
reward_q = np.mean(reward_list_q,axis = 0)
const_q = np.mean(const_list_q,axis = 0)

for i in reward_q:
    f.write(str(i) + '\n')
f.close()

for i in const_q:   
    f2.write(str(i) + '\n')
f2.close()

#reward_ppo_ac = np.mean(reward_list_ppo_ac , axis = 0)
#reward_ppo_ca = np.mean(reward_list_ppo_ac, axis = 0)

#stdr1 = np.std(reward_list_ac,axis = 0)
#stdr2 = np.std(reward_list_ca,axis = 0)
stdr3 = np.std(reward_list_q,axis = 0)
stdr4 = np.std(const_list_q,axis = 0)
#stdr4 = np.std(reward_list_ppo_ac ,axis = 0)
#stdr5 = np.std(reward_list_ppo_ca , axis = 0)

for i in stdr3:
    f1.write(str(i) + '\n')
f1.close()


for i in stdr4:
    f3.write(str(i) + '\n')
f3.close()

#print('avg_reward_ac=',reward_ac[N-1]) 
#print('avg_reward_ca=',reward_ca[N-1])
print('avg_reward_q=',reward_q[N-1])
'''print('avg_reward_ppo_ac=',reward_ppo_ac[N-1])
print('avg_reward_ppo_ca=',reward_ppo_ca[N-1])'''
    
#print('sdt_reward_ac =',stdr1[N-1]) 
#print('sdt_reward_ca = ',stdr2[N-1])
print('sdt_reward_q =',stdr3[N-1]) 
'''print('sdt_reward_ppo_ac= ',stdr4[N-1])
print('sdt_reward_ppo_ca =',stdr5[N-1]) '''




