import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
class DQNBaseAgent:
    def __init__(self,input_dims,n_actions,fc1,
                 fc2,alpha,gamma,tau):
        self.input_dims = input_dims
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions

        self.eval_net = Net(alpha,input_dims,fc1,fc2,n_actions)
        self.target_net = Net(alpha,input_dims,fc1,fc2,n_actions)

        self.update_network_parameters(tau=1.0)

    def choose_action(self,observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.eval_net.device)
        actions = self.eval_net.forward(state)
        return actions.detach().cpu().numpy()[0]

    def update_network_parameters(self, tau=None):
        target_params = self.target_net.named_parameters()
        eval_params = self.eval_net.named_parameters()

        target_state_dict = dict(target_params)
        eval_state_dict = dict(eval_params)
        for name in eval_state_dict:
            eval_state_dict[name] = tau * eval_state_dict[name].clone() + \
                                       (1 - tau) * target_state_dict[name].clone()

        self.target_net.load_state_dict(eval_state_dict)




class Cache_MADQN:
    def __init__(self,s_dims,n_actions,n_agents,fc1,
                 fc2,alpha,gamma,tau):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.input_dims = s_dims
        self.gamma = gamma
        for agent_idx in range(self.n_agents):
            self.agents.append(DQNBaseAgent(self.input_dims,n_actions,fc1,fc2,alpha,gamma,tau))

    def choose_action(self, raw_obs):
        state = []
        for list in raw_obs:
            state += list
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(state)
            actions.append(action)
        return actions

    def learn(self,memory,):
        device = self.agents[ 0 ].eval_net.device
        states, actions, rewards, new_states = memory.sample_buffer()

        for agent_idx, agent in enumerate(self.agents):
            agent_state = torch.tensor(states, dtype=torch.float32).to(device)
            agent_action = torch.tensor(actions[ agent_idx ], dtype=torch.float32).to(device)
            agent_reward = torch.tensor(rewards[ agent_idx ], dtype=torch.float32).to(device)
            agent_new_state = torch.tensor(new_states, dtype=torch.float32).to(device)

            with torch.no_grad():
                q_ = agent.target_net.forward(agent_new_state)
                target = agent_reward + self.gamma * q_.detach().max(0)[0]
            q = agent.eval_net.forward(agent_state)

            loss = F.mse_loss(q,target)

            agent.eval_net.zero_grad()
            loss.backward()
            agent.eval_net.optimizer.step()

            agent.update_network_parameters(tau=agent.tau)

class Net(nn.Module):
    def __init__(self,alpha,input_dims,fc1_dims,fc2_dims,n_actions):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(input_dims,fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        #通过Gumbel-Softmax Trick连续化
        pi = self.pi(x)
        return pi



