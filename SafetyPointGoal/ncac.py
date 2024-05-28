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
import random as rnd
import safety_gymnasium as sg
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
        x = torch.tanh(x)
        x = self.l2(x)
        return x
    
class policyparameter(nn.Module):
    def __init__(self,n1,n2,n3):
        super(policyparameter,self).__init__()
        self.l1 = nn.Linear(n1,n2)
        self.l2 = nn.Linear(n2,n3)
        
    def forward(self,x):
         x =self.l1(x)
         x = torch.tanh(x)
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
        x = torch.tanh(x)
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

def invert_FIM(FIM):
    # Compute the inverse of the FIM
    FIM += torch.eye(FIM.shape[0]) * 1e-5
    inv_FIM = torch.inverse(FIM)

    return inv_FIM


def train_ac():
    #np.random.seed(seed)
    value = valuefunction(d1,d2,d3)
    policy = policyparameter(d1,d2,nA)
    voptim = torch.optim.SGD(value.parameters(),lr = 1.5)
    # poptim = torch.optim.SGD(policy.parameters(),lr = 1.5)
    lambda1 = lambda epoch : (1 + epoch)**(-0.4)
    # lambda2 = lambda epoch : (1 + epoch)**(-0.6)  
    vscheduler = LambdaLR(voptim,lambda1)
    # pscheduler = LambdaLR(poptim,lambda2)
    state ,info = env.reset()
    J = 0
    const = 0
    

    n = 1

    L = 0
    Y = 0
    gamma = 0
    alpha = 0.1
    p = 10000000
    G = 1
    FIM_1 = torch.eye(600)*10
    FIM_2 = torch.eye(10)*10
    FIM_3 = torch.eye(40)*10
    FIM_4 = torch.eye(4)*10
    FIM = [FIM_1, FIM_2, FIM_3, FIM_4]
    prev_FIM = []
    while n <= n_train:

        #Actor critic learning------------
        probs=policy(feat(state))
        action = Categorical(probs).sample()
        action_ = getaction(action)
        next_state,reward,cost ,terminated,truncated ,info= env.step(action_)
        #print(cost)
        if(terminated or truncated):
            next_state,info = env.reset()

        J = (J + reward)/n #average reward
        #returns.append(reward)
        #reward_list.append(np.mean(returns))
        #print(J)
        a = 1.5/(n**0.4)
        c = 1.5/(n**0.8)
        b = 1.5/(n**0.6)

        delta=reward - gamma*(cost - alpha) - L + value(feat(next_state)).detach()-value(feat(state))
        #delta=reward  + 0.9*value(feat(next_state)).detach()-value(feat(state))
        L += a*(reward - gamma*(cost - alpha)-L)
        vloss=0.5*delta**2
        aa = Categorical(probs).log_prob(action)
        fi = aa*aa
        # G = G + a*(fi - G)
        # ploss=-(delta.detach())/Categorical(probs).log_prob(action)
        log_prob = torch.log(policy(feat(state))[action])

        # Compute the gradient of the log probability
        log_prob_grad = torch.autograd.grad(log_prob, policy.parameters(), create_graph=True)


        

        
        i =0
        cc = 0
        for gradd, param in zip(log_prob_grad, policy.parameters()):
            
            # print("Before Update : ", param)
            
            
            
            mm = 0
            if(gradd.dim() == 1):
                nn = gradd.shape
            else: 
                nn, mm = gradd.shape
            #print("Shape: ", n,m)
            gradd = gradd.view(-1,1)
            #print("Shape after view: ", gradd.shape)
            if(n==1):
               FIM = torch.mm(gradd, gradd.t())
               prev_FIM.append(FIM)
            else:
                print(cc)
                FIM = (1 - a)*prev_FIM[cc] + a*torch.mm(gradd, gradd.t())
                prev_FIM[cc] = FIM
            cc = cc + 1
            #print("Shape after mm: ", FIM.shape)
            change = torch.mm(invert_FIM(FIM), gradd)

            #print("Shape after inverse: ", change.shape)
            
            if(mm == 0):
                change = change.view(nn)
            else:
                change = change.view(nn, mm)

            #z = param.view(-1, 1)
            #### Update params 
            # print("---before")
            # print(param.data)
            # print('delta' , delta.shape)
            # print('param_data' , param.data.shape)
            # print ('out' , (b * delta * change).shape) 

            # aa = param.data
            # bb = b * delta * change  
            # cc = aa + bb    
            param.data = param.data + b* delta * change
            #i +=1 
        Y = Y + a*(cost - Y)
        gamma = gamma = max(0, min(p, gamma + c * (Y - alpha)))
        #print(vloss)
        #print(ploss.shape)
        voptim.zero_grad()
        vloss.backward()
        voptim.step()
        # poptim.zero_grad()
        # ploss.backward(retain_graph=True)
        # poptim.step()
        vscheduler.step()
        # pscheduler.step()


        n += 1
        #print(n)
        #print('state= ', state, 'action = ',int(action),'next state = ',next_state)
        state = next_state
    return policy

def actor_critic(seed):  
    np.random.seed(seed) 
    policy = train_ac()
    state ,info = env.reset()
    #print(state)
    indices = []
    reward_list =[]
    const_list =[]
    returns = deque(maxlen = N)
    returns_c = deque(maxlen = N)     
    m = 1
    while(m <= N):
        probs=policy(feat(state))
        action = Categorical(probs).sample()
        action_ = getaction(action)
        next_state,reward,cost ,terminated,truncated ,info= env.step(action_)
        #print(cost)
        if(terminated or truncated):
            next_state,info = env.reset()
        returns.append(reward)
        returns_c.append(cost)
        indices.append(m)
        reward_list.append(np.mean(returns))
        const_list.append(np.mean(returns_c))
        
        m += 1
        state = next_state 
    return reward_list,const_list ,indices
        



  
       

f = open("plotting_nac_final.txt", "a")
f1 = open("plotting_nac_sdt_final.txt", "a")
f2 = open("plotting_nac_const_final.txt", "a")
f3 = open("plotting_nac_const_std_final.txt", "a")
#f2 = open("plotting_ac_min.txt", "a")
n_seed = 10
seed = randint(1000,size = (n_seed,1))
policy= train_ac()
for i in range(0,n_seed):
      seed[i] = randint(1000)
reward_list_ac = np.zeros((n_seed,N))
const_list_ac = np.zeros((n_seed,N))

    
for i in range(0,n_seed):
      #reward_list_q[i] , indices = DQN(seed[i][0])
      reward_list_ac[i],const_list_ac[i], indices = actor_critic(seed[i][0])
      #reward_list_ca[i],indices = critic_actor(seed[i][0])
      #indices , reward_list_ppo_ac[i] = PPO_actor_critic(seed[i][0])
      #indices , reward_list_ppo_ca[i] = PPO_critic_actor(seed[i][0])
      #print(i)
reward_ac = np.mean(reward_list_ac,axis = 0)
const_ac = np.mean(const_list_ac,axis = 0)

for i in reward_ac:
    f.write(str(i) + '\n')
f.close() 

for i in const_ac:
    f2.write(str(i) + '\n')
f2.close()


#reward_ca = np.mean(reward_list_ca,axis = 0)
#reward_q = np.mean(reward_list_q,axis = 0)
#reward_ppo_ac = np.mean(reward_list_ppo_ac , axis = 0)
#reward_ppo_ca = np.mean(reward_list_ppo_ac, axis = 0)

stdr1 = np.std(reward_list_ac,axis = 0)
stdr2 = np.std(const_list_ac,axis = 0)
#stdr2 = np.std(reward_list_ca,axis = 0)
#stdr3 = np.std(reward_list_q,axis = 0)
#stdr4 = np.std(reward_list_ppo_ac ,axis = 0)
#stdr5 = np.std(reward_list_ppo_ca , axis = 0)


for i in stdr1:
    f1.write(str(i) + '\n')
f1.close()

for i in stdr2:
    f3.write(str(i) + '\n') 
f3.close()



print('avg_reward_ac=',reward_ac[N-1]) 
'''print('avg_reward_ca=',reward_ca[N-1])
print('avg_reward_q=',reward_q[N-1])
print('avg_reward_ppo_ac=',reward_ppo_ac[N-1])
print('avg_reward_ppo_ca=',reward_ppo_ca[N-1])'''
    
print('sdt_reward_ac =',stdr1[N-1]) 
'''print('sdt_reward_ca = ',stdr2[N-1])
print('sdt_reward_q =',stdr3[N-1]) 
print('sdt_reward_ppo_ac= ',stdr4[N-1])
print('sdt_reward_ppo_ca =',stdr5[N-1]) '''



       
