from PaperEnv.Contents import Contents
from PaperEnv.Cache_Env import Env
from PaperEnv.cache_ddpg import cache_ddpg
from PaperEnv.cache_RelayBuffer import DDPGRelayBuffer
from PaperEnv import utils
import numpy as np


def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

def execute(fc1,fc2,alpha,beta,gama,tau,zipf,content_num,capacity,n_agent,slot,name):
    contents = Contents(content_num, zipf)
    env = Env(contents, name, capacity, n_agent, slot)
    MAX_EPISODE = 102400
    # 定义Actor和Critic网络的维度
    actor_dims = content_num * 3  # 每个智能体的Actor网络维度一样
    n_actions = content_num  # 动作的维度
    ddpg = cache_ddpg(actor_dims, n_actions, n_agent,
                                 fc1, fc2, alpha, beta, gama, tau)

    memory = DDPGRelayBuffer(20000, actor_dims, n_actions, n_agent, batch_size=64)
    obs = env.reset()
    episode_step = 1
    episode_sum_reward = 0

    reward_list = [ ]
    hite_rate_list = [ ]
    trans_size_list = [ ]
    cache_profit_list = [ ]
    for i in range(MAX_EPISODE):
        actions = ddpg.choose_action(obs)

        obs_, rewards,actions = env.step(actions)

        memory.store_transition(obs, actions, rewards, obs_)


        episode_sum_reward += sum(rewards)

        if (episode_step % 1024 == 0):
            reward_list.append([ episode_sum_reward / 1024 ])
            hitrate, trans_size = utils.getTargetInfo(env)
            # hit_rate.append(episode_hit_rate)
            hite_rate_list.append([ hitrate ])
            trans_size_list.append([ trans_size / 1024 / n_agent ])
            cache_profit_list.append([ env.cache_profit ])
            print("episode avg reward:", episode_sum_reward / 1024)
            print("hit_rate:", hitrate)
            print(trans_size / 1024 / n_agent)
            print("cache_profit", env.cache_profit)


            episode_sum_reward = 0
            ddpg.learn(memory)
        env.cache_profit = 0
        episode_step += 1
        obs = obs_
    cache_profit_filename = "datas/redundancy_datas/" + name + "_redundancy_zipf-" + str(zipf)[ 2 ] + "_size-" + str(
        capacity) + ".xls"
    # reward_filename = "datas/rewards_datas/" + name + "_rewards_zipf" + str(zipf)[2] + "_size" + str(capacity) + ".xls"
    # hitrate_filename = "datas/hitrate_datas/" + name + "_hite_zipf" + str(zipf)[2] + "_size" + str(capacity) + ".xls"
    # trans_size_filename = "datas/trans_size_datas/"+ name + "_trans_zipf" + str(zipf)[2] + "_size" + str(capacity) + ".xls"
    # utils.writeToExcel(reward_list, reward_filename, [ "reward" ])
    # utils.writeToExcel(hite_rate_list, hitrate_filename, [ "hite_rate" ])
    # utils.writeToExcel(trans_size_list, trans_size_filename, [ "trans_size" ])
    utils.writeToExcel(cache_profit_list, cache_profit_filename, [ 'cache_profit' ])

if __name__ == '__main__':
    fc1 = 128  # 第一层神经元个数
    fc2 = 128  # 第二层神经元个数
    alpha = 0.03  # Actor学习率
    beta = 0.03  # Critic学习率
    gama = 0.95  # 折扣因子
    tau = 0.01  # 滑动更新参数
    zipf_parameter = 0.5  # zipf分布的参数
    content_num = 2000  # 内容数量
    base_capacity = 200  # 基站容量
    n_agent = 5
    slot = 60
    contents = Contents(content_num, zipf_parameter)
    execute(fc1, fc2, alpha, beta, gama, tau, zipf_parameter, content_num, base_capacity, n_agent, slot)









