import math
import numpy as np

class Env:
    def __init__(self, contents,flag,capacity,n_agents,slot):
        self.n = n_agents  # 智能体个数
        self.flag = flag
        self.contents = contents
        self.contents_num = contents.content_num
        self.basestations = []#初始化每个基站
        self.capacity = capacity
        for i in range(self.n):
            # basestations为一个数组，数组中的每个元素是基站id和基站BaseStation的一个实例对象构成的数组
            self.basestations.append(BaseStation(self.contents,self.capacity))
        # 定义三种情况下的传输速率
        self.local_tramission_rate = 10#M/s后续计算
        self.near_tramission_rate = 8#M/s
        self.romote_tramission_rate = 5#M/s
        # 定义基站带宽
        self.bandwith = 200  # 带宽
        self.SINR = 40  # 信噪比
        self.slot = slot # 时隙
        self.local_cached = set()  # 记录已经命中的请求
        self.neard_cached = set() #记录附近基站已经命中的请求
        self.remote_cached = set()#记录从云端获取的
        self.EPSILON = 0.9
        self.system_cacheed_redundancy = [0] * contents.content_num  # 系统内容的冗余度
        self.cache_profit = 0

    def calculate_redundancy(self):
        self.system_cacheed_redundancy = [ 0 ] * self.contents.content_num
        for base in self.basestations:
            for contend_id in base.cache_status:
                self.system_cacheed_redundancy[ contend_id ] += 1
        for i in range(self.contents.content_num):
            self.system_cacheed_redundancy[ i ] /= self.n


    # 定义函数附近基站是否缓存了id为content_id的内容
    def if_exist_content(self, contend_id):
        for base in self.basestations:
            if (contend_id in base.cache_status):
                return True
        return False

    # 定义环境的reset函数，初始化环境，返回初始的状态
    def reset(self):
        observation = []
        for i in range(self.n):
            base = self.basestations[i]
            # 使用香农公式计算基站本地的传输速率
            self.local_tramission_rate = self.bandwith / base.user_nums * math.log2(1 + self.SINR)
            # 每个基站开始处理内容请求
            one_slot_time = 0
            while(one_slot_time < self.slot):
                request_id = np.random.choice(self.contents.content_num, 1, p=self.contents.zipfprobality)[0]
                base.request_status[request_id] = 1
                current_content_size = self.contents.content_size_arr[request_id]
                if(request_id in base.cache_status):
                    if(request_id not in self.local_cached):
                        # 第一次命中，第二次命中不计算时延，因为可以通过广播的方式传输命中的数据，计算本地处理的时延
                        one_request_time = current_content_size / (self.local_tramission_rate)
                        self.cache_profit += (1 / one_request_time) * math.exp(math.sqrt(
                            current_content_size / base.cache_capacity * self.system_cacheed_redundancy[ request_id ]))
                        self.local_cached.add(request_id)#第一次命中时添加
                        one_slot_time += one_request_time
                elif(self.if_exist_content(request_id)):
                    if(request_id not in self.neard_cached):
                        # 第一次命中，，第二次命中不计算时延，因为可以通过广播的方式传输命中的数据，计算本地处理的时延
                        # 被附近基站处理的时延
                        one_request_time = current_content_size / self.near_tramission_rate + current_content_size / (self.local_tramission_rate)
                        self.cache_profit += (1 / one_request_time) * math.exp(math.sqrt(
                            current_content_size / base.cache_capacity * self.system_cacheed_redundancy[ request_id ]))
                        self.neard_cached.add(request_id)
                        one_slot_time += one_request_time
                else:
                    if(request_id not in self.remote_cached):
                        one_request_time = current_content_size / self.romote_tramission_rate + current_content_size / (self.local_tramission_rate)
                        self.cache_profit += (1 / one_request_time) * math.exp(math.sqrt(
                            current_content_size / base.cache_capacity * self.system_cacheed_redundancy[ request_id ]))
                        self.remote_cached.add(request_id)
                        one_slot_time += one_request_time

            # 记录每个基站每个时隙内处理的请求
            # base.content_tramissionednum_perstep.append(one_slot_tramissioned)
            # base.hit_num_perstep.append(one_slot_hit_num)

            # 清空记录以便下个基站记录处理请求
            self.local_cached.clear()
            self.neard_cached.clear()
            self.remote_cached.clear()

        # 添加内容的冗余度
        self.caluate_redundancy()
        # observation.append(self.system_cacheed_redundancy)

        for base in self.basestations:

            # 基站存储内容大小(与基站的存储状态相对应)
            # 基站存储内容流行度(与基站的存储状态相对应)
            #基站内容的冗余度

            obs_arr = base.request_status + base.cached_content_size + base.cached_content_prob
            observation.append(obs_arr)

        #print(observation)
        return observation

    def step(self, actions):
        agents_rewards = []
        new_actions = []#将得分转换为具体的动作
        for agent_index in range(self.n):  # 一次处理一个基站的请求
            base = self.basestations[agent_index]
            base_action = []
            new_action = [0] * self.contents_num
            ###对输出动作的得分进行排序
            for i in range(self.contents_num):
                base_action.append([i,actions[agent_index][i]])
            base_action.sort(key=lambda ele:ele[1],reverse=True)#得分高的排在前面，类似[[0.5,3],[0.3,1]....]


            for content_id in base.cache_status:
                base.tmp_set.add(content_id)
            base.cache_status.clear()
            base.cache_capacity = 0

            index = 0#内容得分从高往低加入缓存空间
            while(base.cache_capacity < base.max_cache_capacity):

                if(np.random.uniform() < self.EPSILON):
                    content_id =  base_action[index][0]
                    index += 1
                else:
                    content_id = np.random.choice(self.contents_num,1)[0]
                if(content_id not in base.cache_status):
                    new_action[content_id] = 1
                    base.cache_status.add(content_id)
                    base.cache_capacity += self.contents.content_size_arr[content_id]
            new_actions.append(new_action)

        self.caluate_redundancy()
        for i in range(self.n):
            # 产生下一个时隙的内容请求
            base = self.basestations[i]
            base.request_status = [0] * self.contents_num#清除上一个时隙的请求状态
            agent_reward = 0 # 定义单个基站的奖励
            one_slot_time = 0  # 定义处理内容请求的时间

            while(one_slot_time < self.slot):
                request_id = np.random.choice(self.contents_num,1,p=self.contents.zipfprobality)[0]
                current_content_size = self.contents.content_size_arr[request_id]
                #if(request_id in base.cache_status ):#and request_id not in base.tmp_set

                base.request_status[request_id] = 1
                base.request_num += 1
                if (request_id in base.cache_status):
                    one_request_time = current_content_size / (self.local_tramission_rate)
                    self.cache_profit += (1 / one_request_time) * math.exp(math.sqrt(
                        current_content_size / base.cache_capacity * self.system_cacheed_redundancy[ request_id ]))
                    if (request_id not in self.local_cached):
                        # 第一次命中，第二次命中不计算时延，因为可以通过广播的方式传输命中的数据，计算本地处理的时延

                        self.local_cached.add(request_id)  # 第一次命中时添加
                        one_slot_time += one_request_time
                        agent_reward += math.pow(self.contents.content_size_arr[ request_id ],
                                                 1 - self.system_cacheed_redundancy[ request_id ])
                        base.transmission_size += current_content_size
                    base.hit_num += 1


                elif (self.if_exist_content(request_id)):
                    if (request_id not in self.neard_cached):
                        # 第一次命中，，第二次命中不计算时延，因为可以通过广播的方式传输命中的数据，计算本地处理的时延
                        # 被附近基站处理的时延
                        if (request_id in base.tmp_set):
                            agent_reward -= math.pow(self.contents.content_size_arr[ request_id ],
                                                     1 - self.system_cacheed_redundancy[ request_id ])
                        one_request_time = current_content_size / self.near_tramission_rate + current_content_size / (
                                    self.local_tramission_rate)
                        self.cache_profit += (1 / one_request_time) * math.exp(math.sqrt(
                            current_content_size / base.cache_capacity * self.system_cacheed_redundancy[ request_id ]))
                        self.neard_cached.add(request_id)

                        base.transmission_size += current_content_size
                        one_slot_time += one_request_time
                else:
                    if (request_id not in self.remote_cached):
                        one_request_time = current_content_size / self.romote_tramission_rate + current_content_size / (
                                    self.local_tramission_rate)
                        if (request_id in base.tmp_set):
                            agent_reward -= math.pow(self.contents.content_size_arr[ request_id ],
                                                     1 - self.system_cacheed_redundancy[ request_id ])
                        self.cache_profit += (1 / one_request_time) * math.exp(math.sqrt(
                            current_content_size / base.cache_capacity * self.system_cacheed_redundancy[ request_id ]))
                        self.neard_cached.add(request_id)
                        one_slot_time += one_request_time
                        base.transmission_size += current_content_size
            base.tmp_set.clear()
            self.local_cached.clear()
            self.neard_cached.clear()
            self.remote_cached.clear()
            agents_rewards.append(agent_reward)

        observation = []
        for base in self.basestations:
            # 基站存储内容大小(与基站的存储状态相对应)
            # 基站存储内容流行度(与基站的存储状态相对应)
            # 基站内容的冗余度
            for i in range(self.contents_num):
                if(i not in base.cache_status):
                    base.cached_content_prob[i] = 0
                    base.cached_content_size[i] = 0
                else:
                    base.cached_content_prob[ i ] = self.contents.content_size_arr[i]
                    base.cached_content_size[ i ] = self.contents.all_content_prob[i]
            observation.append(base.request_status + base.cached_content_size + base.cached_content_prob)
        return observation,agents_rewards,new_actions


    # 将基站的状态从列表转换成array类型


    # 计算冗余度
    def caluate_redundancy(self):
        self.system_cacheed_redundancy = [0] * self.contents_num
        length = self.contents_num
        for i in range(length):
            for base in self.basestations:
                if (i in base.cache_status):
                    self.system_cacheed_redundancy[i] += 1
        self.system_cacheed_redundancy = np.array(self.system_cacheed_redundancy)
        self.system_cacheed_redundancy = self.system_cacheed_redundancy / self.n
        self.system_cacheed_redundancy = list(self.system_cacheed_redundancy)

class BaseStation:
    def __init__(self, contents,capacity):
        self.content_num = contents.content_num
        self.cache_status = set()  # 定义基站的内容存储状态
        self.tmp_set = set()#记录上一个时隙基站的存储状态
        self.cached_content_prob = [0] * contents.content_num  # 记录存储内容的流行度，长度固定
        self.cached_content_size = [0] * contents.content_num  # 记录存储内容的大小，长度固定
        self.request_status = [0] * contents.content_num

        self.cached_content_redundancy = [0] * contents.content_num  # 记录存储内容的冗余度，长度固定
        self.max_cache_capacity = capacity # 定义基站的最大内容存储空间
        self.hit_num = 0
        self.request_num = 0
        self.transmission_size = 0
        self.cache_capacity = 0  # 定义基站存储内容所占的空间大小

        while (True):
            tmp_id = np.random.choice(contents.content_num,1)[0]
            if (tmp_id not in self.cache_status):
                if (self.cache_capacity + contents.content_size_arr[tmp_id] > self.max_cache_capacity):
                    break
                self.cache_status.add(tmp_id)
                self.cached_content_prob[tmp_id] = contents.all_content_prob[tmp_id]
                self.cached_content_size[tmp_id] = contents.content_size_arr[tmp_id]
                self.cache_capacity += contents.content_size_arr[tmp_id]

        # 初始化请求队列,打算设置时隙为1s，基站的传输速度在50M/s，有线传输的速度大概在100M/s
        # 将每个用户产生的请求添加到请求队列中

