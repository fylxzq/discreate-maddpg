from PaperEnv.Contents import Contents
import numpy as np
import math
from PaperEnv import utils
class FIFO:
    def __init__(self,contents,capacity,slot,n_agent):
        self.contents = contents
        self.base_num = n_agent
        self.basestations = []
        self.zipf = contents.zipf_beta
        self.capacity = capacity

        # 定义三种情况下的传输速率
        self.local_tramission_rate = 10  # 后续计算
        self.near_tramission_rate = 8 # M/s
        self.romote_tramission_rate = 5 # M/s
        self.slot = slot # 时隙
        self.system_cacheed_redundancy = [0] * contents.content_num
        self.local_cached = set()  # 记录已经命中的请求
        self.neard_cached = set()  # 记录附近基站已经命中的请求
        self.remote_cached = set()  # 记录从云端获取的

        for i in range(self.base_num):
            self.basestations.append(FIFOBaseStations(self.contents,capacity))

        self.calculate_redundancy()
        self.cache_profit = 0

    def calculate_redundancy(self):
        self.system_cacheed_redundancy = [ 0 ] * self.contents.content_num
        for base in self.basestations:
            for contend_id in base.cache_status_set:
                self.system_cacheed_redundancy[ contend_id ] += 1
        for i in range(self.contents.content_num):
            self.system_cacheed_redundancy[ i ] /= self.base_num


    def step(self,fileName,step):
        slot_times = 0
        redundancy_list = [ ]
        while (slot_times < 50):

            for base in self.basestations:
                one_time_perstep = 0
                miss_list = []
                while(one_time_perstep < self.slot):
                    request_id = np.random.choice(self.contents.content_num,1,p=self.contents.zipfprobality)[0]
                    current_content_size = base.contents.content_size_arr[request_id]
                    self.local_tramission_rate = self.bandwith / base.user_nums * math.log2(1 + self.SINR)
                    if(request_id in base.cache_status_set):
                        base.hit_num += 1

                        one_request_time = current_content_size / (self.local_tramission_rate)
                        self.cache_profit += (1 / one_request_time) * math.exp(math.sqrt(current_content_size / base.cache_capacity * self.system_cacheed_redundancy[request_id]))
                        if (request_id in self.local_cached):  # 如果当前请求内容已在该时隙被处理过
                            continue
                        # 该时隙第一次命中
                        if (one_request_time + one_time_perstep  > self.slot):  # 大于一个时隙，退出循环，不计入处理个数
                            #一个时隙结束，重置大部分辅助数据
                            self.local_cached.clear()
                            self.neard_cached.clear()
                            self.remote_cached.clear()
                            break
                        base.total_size += current_content_size
                        self.local_cached.add(request_id)
                        one_time_perstep += one_request_time
                    elif(self.if_exist_content(request_id)):
                        one_request_time = current_content_size / self.near_tramission_rate + current_content_size / (
                                    self.local_tramission_rate)
                        self.cache_profit += (1 / one_request_time) * math.exp(math.sqrt(current_content_size / base.cache_capacity * self.system_cacheed_redundancy[request_id]))


                        if (request_id in self.neard_cached):
                            continue
                        if (one_request_time + one_time_perstep  > self.slot):  # 大于一个时隙，退出循环，不计入处理个数
                            # 一个时隙结束，重置大部分辅助数据
                            one_time_perstep = 0
                            self.local_cached.clear()
                            self.neard_cached.clear()
                            self.remote_cached.clear()
                            break
                        base.total_size += current_content_size
                        self.neard_cached.add(request_id)
                        one_time_perstep += one_request_time
                        miss_list.insert(0,request_id)

                    else:
                        one_request_time = current_content_size / self.romote_tramission_rate + current_content_size / (
                            self.local_tramission_rate)
                        self.cache_profit += (1 / one_request_time) * math.exp(math.sqrt(current_content_size / base.cache_capacity * self.system_cacheed_redundancy[request_id]))
                        if(request_id in self.remote_cached):
                            continue
                        if (one_request_time + one_time_perstep > self.slot):
                            # 一个时隙结束，重置大部分辅助数据
                            one_time_perstep = 0
                            self.local_cached.clear()
                            self.neard_cached.clear()
                            self.remote_cached.clear()
                            break
                        base.total_size += current_content_size
                        one_time_perstep += one_request_time
                        self.remote_cached.add(request_id)
                        miss_list.insert(0,request_id)
                        # 执行先进先出的缓存替换策略
                    base.request_num += 1


                while len(miss_list) > 0:
                    request_id = miss_list.pop()
                    current_content_size = self.contents.content_size_arr[request_id]
                    while(request_id not in base.cache_status_set and base.max_cache_capacity - base.cache_capacity < current_content_size):
                        remove_id = base.cache_status_queue.pop()
                        base.cache_status_set.remove(remove_id)
                        base.cache_capacity -= self.contents.content_size_arr[request_id]
                    base.cache_status_set.add(request_id)
                    base.cache_status_queue.push(request_id)
                    base.cache_capacity += current_content_size

            redundancy_list.append(self.cache_profit)
            self.cache_profit = 0
            slot_times += 1

            # 计算冗余度

        # print(redundancy_list)
        utils.writeRedundancy(redundancy_list,step,fileName)

        #result = self.handle_redundancy()
        all_basestation_hit_nums = 0
        all_basestation_request_nums = 0
        all_basestation_totoal_size = 0
        for base in self.basestations:
            all_basestation_hit_nums += base.hit_num
            all_basestation_request_nums += base.request_num
            all_basestation_totoal_size += base.total_size
            base.hit_num = 0
            base.request_num = 0
            base.total_size = 0

        hit_rate = all_basestation_hit_nums / all_basestation_request_nums
        trans_size = all_basestation_totoal_size / 1024 / self.base_num
        print(hit_rate)
        print(trans_size)

        return hit_rate,trans_size



    def if_exist_content(self,request_id):
        for base in self.basestations:
            if(request_id in base.cache_status_set):
                return True

        return False


class FIFOBaseStations:
    def __init__(self,contents,capacity):
        self.contents = contents
        self.request_queue = []
        self.content_num = contents.content_num
        self.cache_status_queue  = MyQueue()  # 定义基站的缓存状态，用先进先出的队列表示
        self.cache_status_set = set()#使用一个set辅助表示基站的存储状态
        self.max_cache_capacity = capacity  # 定义基站的最大内容存储空间
        self.cache_capacity = 0  # 定义基站存储内容所占的空间大小
        self.hit_num = 0
        self.request_num = 0
        self.total_size = 0

        #初始化基站的存储状态
        while(True):
            tmp_id = np.random.choice(self.content_num,1)[0]
            if(tmp_id not in self.cache_status_set):
                if (self.cache_capacity + contents.content_size_arr[tmp_id]> self.max_cache_capacity):
                    break
                self.cache_capacity += contents.content_size_arr[tmp_id]
                self.cache_status_queue.push(tmp_id)
                self.cache_status_set.add(tmp_id)


class MyQueue:
    def __init__(self):
        self.queue = []
        self.size = 0

    def push(self,elemet):
        self.queue.insert(0,elemet)
        self.size += 1

    def pop(self):
        if(self.size == 0):
            print("空队列")
            return
        element = self.queue.pop()
        self.size -= 1
        return element

    def get(self):
        return self.queue[self.size-1]


def execute(content_num, zipf, capacity, n_agents, slot, name):
    hite_rate_list = [ ]
    trans_size_list = [ ]
    redundancy_filename = "datas/redundancy_datas/" + name + "_redundancy_zipf-" + str(zipf)[ 2 ] + "_size-" + str(
        capacity) + ".xls"

    for i in range(50):
        contents = Contents(content_num, zipf)
        fifo = FIFO(contents, capacity, slot, n_agents)
        hit_rate, trans_size = fifo.step(redundancy_filename,i)
        hite_rate_list.append([ hit_rate ])
        trans_size_list.append([ trans_size ])

if __name__ == '__main__':
    content_num = 2000
    # zipf = 0.5
    # contents = Contents(content_num,zipf)
    slot = 60
    # capacity = 200
    n_agent = 5
    for zipf in [0.5,0.7,0.9]:
        for capacity in [300]:
            execute(content_num,zipf,capacity,n_agent,slot,"fifo")




