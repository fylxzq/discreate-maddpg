import random
import math
class Contents:
    def __init__(self,content_num,zipf_beta):
        self.content_num = content_num#内容的个数
        self.content_size_arr = []#内容大小数组，数组长度为n
        # zipf分布
        self.all_content_prob = []
        self.zipfprobality = []  # 记录概率
        self.zipf_beta = zipf_beta# 参考王晓飞论文
        self.generatezipf()
        # 初始化每个基站的内容存储状态
        self.init_contents()#初始化内容


        # 定义产生zipf的函数

    def generatezipf(self):
        for i in range(self.content_num):
            prob = math.pow(i + 1, -self.zipf_beta)
            self.all_content_prob.append(prob)

        sum_prob = sum(self.all_content_prob)
        for i in self.all_content_prob:
            self.zipfprobality.append(i / sum_prob)

    def init_contents(self):
        # 索引代表内容ID
        for i in range(self.content_num):
            tmp_size = random.randint(1, 10)
            self.content_size_arr.append(tmp_size)






