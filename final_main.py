from PaperEnv import maddpg_main
from PaperEnv import ddpg_main
from PaperEnv import MADQN_Main
if __name__ == '__main__':
    # change_parmeters()
    fc1 = 128  # 第一层神经元个数
    fc2 = 128  # 第二层神经元个数
    alpha = 0.03  # Actor学习率
    beta = 0.03  # Critic学习率
    gama = 0.95  # 折扣因子
    tau = 0.01  # 滑动更新参数
    zipf_parameters = [0.5,0.9]  # zipf分布的参数
    content_num = 2000  # 内容数量
    base_capacitys = [300]  # 基站容量
    n_agent = 5
    slot = 60

    for zipf_parameter in zipf_parameters:
        for base_capacity in base_capacitys:
            maddpg_main.execute(fc1,fc2,alpha,beta,gama,tau,zipf_parameter,content_num,base_capacity,n_agent,slot,"maddpg")
            ddpg_main.execute(fc1,fc2,alpha,beta,gama,tau,zipf_parameter,content_num,base_capacity,n_agent,slot,"ddpg")
            #MADQN_Main.execute(fc1,fc2,alpha,beta,gama,tau,zipf_parameter,content_num,base_capacity,n_agent,slot,"madqn")