import torch as T
from PaperEnv.cache_network import ActorNetWork,CriticNetWork
class BaseAgent:
    def __init__(self,actor_dims,critic_dims,n_actions,n_agents,agent_indx,
                 alpha,beta,fc1,fc2,gamma,tau):
        self.actor_dims = actor_dims
        self.critic_dims = critic_dims
        self.tau = tau
        self.gamma = gamma
        self.n_actions = n_actions
        self.agent_name = "agent_%s" % agent_indx
        #定义每个智能体的Evaluation Network
        self.actor = ActorNetWork(alpha,self.actor_dims,fc1,fc2,n_actions,
                                       name=self.agent_name+'_actor')

        self.critic = CriticNetWork(beta,self.critic_dims,fc1,fc2,n_agents,n_actions
                                    ,name=self.agent_name+"_critic")
        # 定义每个智能体的Target Network
        self.target_actor = ActorNetWork(alpha, self.actor_dims, fc1, fc2, n_actions,
                                       name=self.agent_name + '_actor')
        self.target_critic = CriticNetWork(beta, self.critic_dims, fc1, fc2,n_agents,n_actions,
                                    name=self.agent_name + "_critic")

        self.update_network_parameters(tau=1)#tau=1保证初始化时Evaluation Network 和Traget Network两者的参数一样


    #单个智能体采取动作
    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        actions = self.actor.forward(state)
        return actions.detach().cpu().numpy()[0]


    #更新Target Network参数，采取平滑更新的方式
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                      (1 - tau) * target_critic_state_dict[name].clone()

        # for name in critic_state_dict:
        #     critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
        #             (1-tau)*target_critic_state_dict[name].clone()
        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()