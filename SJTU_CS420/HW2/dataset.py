import numpy as np
import random
import matplotlib.pyplot as plt

class Dataset(object):
    def __init__(self, m_dim = 3, data_num = 1000, data_dim = 10, var = 0.1, mu = 0, class_num = 3, seed = 4):
        '''
        :param m_dim: dimensionality of y
        :param data_num: sample number
        :param data_dim: dimensionality of x
        :param var: variance^2 of  e
        :param mu: mean of x
        :param class_num: number of clusters
        :param seed: random seed
        :param center: center of data
        '''
        # FA_data
        self.m_dim = m_dim
        self.data_num = data_num
        self.dim = data_dim
        self.var = var
        self.mu = mu
        self.data = np.zeros((data_num, data_dim), dtype=np.float32)
        self.yt = np.zeros(m_dim, dtype=np.float32)
        self.et = np.zeros(data_dim, dtype=np.float32)
        self.center = [mu for i in range(data_dim)]
        self.center_y = np.zeros(m_dim, dtype=np.float32)
        self.center_e = np.zeros(data_dim, dtype=np.float32)
        self.A = np.zeros((data_dim, m_dim), dtype=np.float32)

        # SC_data
        self.class_num = class_num
        if data_dim == 2:
            self.centers = [(0, 0), (0, 1), (1, 1), (1, 0)]
        if data_dim == 3:
            self.centers = [(0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
        if data_dim == 4:
            self.centers = [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)]
        if data_dim == 5:
            self.centers = [(0, 1, 0, 1, 0), (1, 0, 1, 0, 1), (1, 1, 0, 0, 1), (1, 0, 0, 1, 1)]
        if data_dim == 6:
            self.centers = [(0, 1, 1, 0, 1, 0), (1, 0, 1, 1, 0, 1), (1, 1, 1, 0, 0, 1), (1, 0, 0, 1, 1, 1)]
        self.data_sc = np.zeros((data_num*class_num, data_dim), dtype=np.float32)
        self.colors = ['red', 'blue']

        random.seed(seed)

    def generate(self):
        '''
        generate data
        :return: None
        '''
        for i in range(self.dim):
            for x in range(self.m_dim):
                self.A[i][x] = random.random()
        sigma_e = random.random() * self.var
        sigma_t = random.random()
        for i in range(self.data_num):
            for x in range(self.dim):
              self.et[x] = np.random.normal(self.center_e[x], sigma_e)
            for k in range(self.m_dim):
              self.yt[k] = np.random.normal(self.center_y[k], sigma_t)
            self.data[i] = np.dot(self.A, self.yt) + self.center + self.et

    def generate_cluster(self):
        '''
        generate clustered data
        :return: None
        '''
        var = [0.2*(i+1) for i in range(self.class_num)]
        for k in range(self.class_num):
            sigma = random.random() * var[k]
            for i in range(self.data_num):
                for x in range(self.dim):
                    self.data_sc[k * self.data_num + i][x] = np.random.normal(self.centers[k][x], sigma)

    def show(self):
        '''
        show the distribution of data
        :return: None
        '''
        plt.figure()

        x = self.data[0: self.data_num - 1, 0]
        y = self.data[0: self.data_num - 1, 1]
        plt.scatter(x, y, s = 10, c = self.colors[0])
        plt.show()

if __name__ == '__main__':
    dataset = Dataset(3, 100, 5)
    dataset.generate()
    dataset.show()