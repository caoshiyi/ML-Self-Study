import numpy as np
import random
import matplotlib.pyplot as plt

class Dataset(object):
    def __init__(self, class_num = 2, data_num = 200, data_dim = 2, seed = 4, center=None):
        '''
        :param class_num: class number
        :param data_num: data number for each class
        :param data_dim: data dimension
        :param seed: random seed
        :param center: cluster center of data
        '''
        self.class_num = class_num
        self.data_num = data_num
        self.dim = data_dim
        self.data = np.zeros((data_num * class_num, data_dim), dtype = np.float32)
        if center == None:
            if data_dim == 2:
                self.centers = [(0, 0), (0, 1), (1, 1), (1, 0)]
            if data_dim == 3:
                self.centers = [(0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
            if data_dim == 4:
                self.centers = [(0, 1, 0, 1), (1, 0, 1, 0), (1, 1, 0, 0), (1, 0, 0, 1)]
            if data_dim == 5:
                self.centers = [(0, 1, 0, 1, 0), (1, 0, 1, 0, 1), (1, 1, 0, 0, 1), (1, 0, 0, 1, 1)]
            if data_dim == 6:
                self.centers = [(0, 1, 1, 0, 1, 0), (1, 0, 1, 1, 0, 1), (1, 1, 1, 0, 0, 1), (1, 0, 0, 1, 1, 1)]
            #self.centers = [(0, 0), (1, 0), (0.5, 0.866)]
             #self.centers = [(0, 1), (0, 0.9), (0, 1.1), (0.1, 1.1)]
        else:
            self.centers = center
        self.colors = ['red', 'blue']

        random.seed(seed)

    def generate(self):
        '''
        generate data
        :return: None
        '''
        for k in range(self.class_num):
            sigma = random.random() * 0.4
            for i in range(self.data_num):
                for x in range(self.dim):
                    self.data[k * self.data_num + i][x] = np.random.normal(self.centers[k][x], sigma)


    def show(self):
        '''
        show the distribution of data
        :return: None
        '''
        plt.figure()
        for i in range(self.class_num):
            x = self.data[i * self.data_num: (i + 1) * self.data_num - 1, 0]
            y = self.data[i * self.data_num: (i + 1) * self.data_num - 1, 1]
            plt.scatter(x, y, s = 10, c = self.colors[i])
        plt.show()

if __name__ == '__main__':
    dataset = Dataset()
    dataset.generate()
    dataset.show()