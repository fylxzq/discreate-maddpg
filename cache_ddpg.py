from PaperEnv.cache_agent import BaseAgent
import torch as T
import torch.nn.functional as F
class cache_ddpg:
    def __init__(self,s_dims,n_actions,n_agents,fc1,
                 fc2,alpha,beta,gamma,tau):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.input_dims = s_dims
        for agent_idx in range(self.n_agents):
            self.agents.append(BaseAgent(self.input_dims,self.input_dims,n_actions,1,agent_idx,alpha,beta,fc1,fc2,gamma,tau))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        return actions

    def learn(self,memory):
        device = self.agents[0].actor.device
        states,actions,rewards,new_states = memory.sample_buffer()

        for agent_idx, agent in enumerate(self.agents):
            agent_state = T.tensor(states[agent_idx],dtype=T.float).to(device)
            agent_action = T.tensor(actions[agent_idx],dtype=T.float).to(device)
            agent_reward = T.tensor(rewards[agent_idx],dtype=T.float).to(device)
            agent_new_state = T.tensor(new_states[agent_idx],dtype=T.float).to(device)

            a = agent.actor.forward(agent_state)
            q = agent.critic.forward(agent_state,a)
            loss_a = -T.mean(q)
            agent.actor.zero_grad()
            loss_a.backward()
            agent.actor.optimizer.step()

            a_ = agent.target_actor.forward(agent_new_state)
            q_ = agent.target_critic.forward(agent_new_state,a_)
            q_target = agent_reward + agent.gamma*q_
            q_v = agent.critic.forward(agent_state,agent_action)
            td_error = F.mse_loss(q_target,q_v)

            agent.critic.zero_grad()
            td_error.backward()
            agent.critic.optimizer.step()

            agent.update_network_parameters()
