import numpy as np


class Cal(object):
    def __init__(self, matrix=None):
        self.matrix = matrix

    def solve(self, num_iter, num_cpu=1):
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

        print(p1_act_his, p2_act_his)


hh = np.array([
    [0, 1, -1],
    [-1, 0, 1],
    [1, -1, 0]
])
tmp = Cal(hh)
tmp.solve(1000)
