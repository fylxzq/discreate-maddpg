import numpy as np
class MultiAgentRelayBuffer:
    def __init__(self,max_size,actor_dims,critic_dims,n_actions,n_agents,batch_size):
        self.mem_size = max_size#经验池最大容量
        self.mem_cntr = 0#计数变量
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.critic_dims = critic_dims
        self.batch_size = batch_size
        self.n_actions = n_actions


        #初始化Critic训练经验池，全局的
        self.state_memory = np.zeros((self.mem_size,self.critic_dims),dtype='float32')
        self.new_state_memory = np.zeros((self.mem_size,self.critic_dims),dtype='float32')
        self.reward_memory = np.zeros((self.mem_size, n_agents),dtype='float32')

        #初始化Actor训练经验池，每个智能体都有
        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []

        for i in range(self.n_agents):#每个智能体一个经验池
            self.actor_state_memory.append(
                np.zeros((self.mem_size, self.actor_dims),dtype='float32'))
            self.actor_new_state_memory.append(
                np.zeros((self.mem_size, self.actor_dims),dtype='float32'))
            self.actor_action_memory.append(
                np.zeros((self.mem_size, self.n_actions),dtype='float32'))


    #经验池存储数据
    def store_transition(self, raw_obs, state, action, reward,
                         raw_obs_, state_):

        #通过self.mem_cntr的值来确定，类似于循环队列的机制
        index = self.mem_cntr % self.mem_size

        #存储Actor经验池
        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]

        #存储Critic经验池
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]

        actor_states = []
        actor_new_states = []
        actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])

        return actor_states, states, actions, rewards, \
               actor_new_states, states_

    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True

class DDPGRelayBuffer:
    def __init__(self,max_size,input_dims,n_actions,n_agents,batch_size):
        self.mem_size = max_size#经验池最大容量
        self.mem_cntr = 0#计数变量
        self.n_agents = n_agents
        self.input_dims  =input_dims
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.states_memory = []
        self.actions_memory = []
        self.reward_memory = []
        self.new_states_memory = []


        self.init_memory()

    def init_memory(self):

        for i in range(self.n_agents):
            self.states_memory.append(np.zeros((self.mem_size,self.input_dims),dtype='float32'))
            self.actions_memory.append(np.zeros((self.mem_size,self.n_actions),dtype='float32'))
            self.reward_memory.append(np.zeros((self.mem_size,1),dtype='float32'))
            self.new_states_memory.append(np.zeros((self.mem_size, self.input_dims),dtype='float32'))



    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = []
        actions = []
        rewards = []
        new_states = []

        for agent_index in range(self.n_agents):
            states.append(self.states_memory[agent_index][batch])
            actions.append(self.actions_memory[agent_index][batch])
            rewards.append(self.reward_memory[agent_index][batch])
            new_states.append(self.new_states_memory[agent_index][batch])
        return states,actions,rewards,new_states

    def store_transition(self,states,actions,rewards,new_states):
        # 通过self.mem_cntr的值来确定，类似于循环队列的机制
        index = self.mem_cntr % self.mem_size
        for agent_index in range(self.n_agents):
            self.states_memory[agent_index][index] = states[agent_index]
            self.actions_memory[agent_index][index] = actions[agent_index]
            self.reward_memory[agent_index][index] = rewards[agent_index]
            self.new_states_memory[agent_index][index] = new_states[agent_index]

        self.mem_cntr += 1


class MADQNRelayBuffer:
    def __init__(self,max_size,input_dims,n_actions,n_agents,batch_size):
        self.mem_size = max_size#经验池最大容量
        self.mem_cntr = 0#计数变量
        self.n_agents = n_agents
        self.input_dims  =input_dims
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.states_memory = []
        self.actions_memory = []
        self.reward_memory = []
        self.new_states_memory = []


        self.init_memory()
    def init_memory(self):
        self.states_memory = np.zeros((self.mem_size, self.input_dims), dtype='float32')
        self.new_states_memory = np.zeros((self.mem_size, self.input_dims), dtype='float32')
        for i in range(self.n_agents):
            self.actions_memory.append(np.zeros((self.mem_size,self.n_actions),dtype='float32'))
            self.reward_memory.append(np.zeros((self.mem_size,1),dtype='float32'))


    def store_transition(self, states, actions, rewards, new_states):
        # 通过self.mem_cntr的值来确定，类似于循环队列的机制
        index = self.mem_cntr % self.mem_size
        state_list = []
        new_state_list = []
        for agent_index in range(self.n_agents):
            state_list += states[agent_index]
            self.actions_memory[ agent_index ][ index ] = actions[ agent_index ]
            self.reward_memory[ agent_index ][ index ] = rewards[ agent_index ]
            new_state_list += states[agent_index]
        self.states_memory[index] = np.array(state_list)
        self.new_states_memory[index] = np.array(state_list)
        self.mem_cntr += 1


    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        actions = []
        rewards = []


        states = self.states_memory[batch]
        new_states = self.new_states_memory[batch]
        for agent_index in range(self.n_agents):
            actions.append(self.actions_memory[agent_index][batch])
            rewards.append(self.reward_memory[agent_index][batch])
        return states,actions,rewards,new_states