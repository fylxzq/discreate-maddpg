import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ActorNetWork(nn.Module):#选择缓存进来的Actor网络
    def __init__(self,alpha,input_dims,fc1_dims,fc2_dims,n_actions,name):
        super(ActorNetWork,self).__init__()
        self.n_actions = n_actions
        #self.chkpt_file = os.path.join(chkpt_dir,name)

        self.fc1 = nn.Linear(input_dims,fc1_dims)
        self.fc2 = nn.Linear(fc1_dims,fc2_dims)
        self.pi = nn.Linear(fc2_dims,n_actions)

        self.optimizer = optim.Adam(self.parameters(),lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        #通过Gumbel-Softmax Trick连续化
        noise = np.random.gumbel(size=self.n_actions)#通过Gi=−log(−log(ϵi))计算得到Gi
        noise = noise.reshape((1,self.n_actions))
        noise = T.tensor(noise,dtype=T.float).to(self.device)
        pi = self.pi(x) + noise
        pi = T.softmax(pi,dim=1)
        pi = pi * 10
        return pi



class CriticNetWork(nn.Module):#构建Critic网络
    def __init__(self,beta,input_dims,fc1_dims,fc2_dims,n_agents,n_actions,name):
        super(CriticNetWork,self).__init__()

        #self.chkpt_file = os.path.join(chkpt_dir,name)

        self.fc1 = nn.Linear(input_dims+n_agents*n_actions,fc1_dims)
        self.fc2 = nn.Linear(fc1_dims,fc2_dims)
        self.q = nn.Linear(fc2_dims,1)

        self.optimizer = optim.Adam(self.parameters(),lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

