import numpy as np
import time
from joblib import Parallel, delayed


class FP(object):
    def __init__(self, matrix=None):
        self.matrix = matrix

    def sub_solve(self, num_iter):
        x_len = self.matrix.shape[0]
        y_len = self.matrix.shape[1]
        p1_value = np.zeros(x_len)
        p2_value = np.zeros(y_len)
        p1_action = np.random.randint(x_len)
        p2_action = np.random.randint(y_len)
        p1_act_his = np.zeros(x_len)
        p2_act_his = np.zeros(y_len)
        i_iter = 0
        while i_iter < num_iter:
            p1_act_his[p1_action] += 1
            p2_act_his[p2_action] += 1
            p1_value += self.matrix[:, p2_action]
            p2_value += self.matrix[p1_action, :]
            p1_action = np.argmax(p1_value)
            p2_action = np.argmin(p2_value)
            i_iter += 1
        return p1_act_his, p2_act_his

    def solve(self, num_iter, num_cpu=1):
        tmp_result = Parallel(n_jobs=num_cpu)(delayed(self.sub_solve)(num_iter) for _ in range(num_cpu))
        p1_policy = np.zeros_like(tmp_result[0][0])
        p2_policy = np.zeros_like(tmp_result[0][0])
        for sub_ans in tmp_result:
            p1_policy += sub_ans[0]
            p2_policy += sub_ans[1]
        p1_policy /= (num_iter * num_cpu)
        p2_policy /= (num_iter * num_cpu)
        return p1_policy, p2_policy


hh = np.array([
    [0, 1, -1],
    [-1, 0, 1],
    [1, -1, 0]
])
tmp = FP(hh)
start = time.time()
policy = tmp.solve(50000, 4)
end = time.time()

print(end - start)
print(policy)
