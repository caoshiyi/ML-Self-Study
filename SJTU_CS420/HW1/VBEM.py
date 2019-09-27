from sklearn.mixture import BayesianGaussianMixture
from dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class VBEM(object):
    def __init__(self, n_components = 1, verbose = 2, verbose_interval = 1,
                 Data = None):
        '''
        :param n_components: cluster number
        :param verbose: whether to show training details
        :param verbose_interval: showing training details interval
        :param Data: dataset
        '''
        self.model = BayesianGaussianMixture(
            n_components = n_components,
            verbose = verbose,
            verbose_interval = verbose_interval)
        self.n_components = n_components
        if Data == None:
            self.dataset = Dataset()
            self.dataset.generate()
        else:
            self.dataset = Data
        self.data = self.dataset.data

    def train(self):
        self.model.fit(self.data)

    def show(self, n = None):
        '''
        show the result of trained model
        :param n: just used for save files
        :return: None
        '''
        plt.figure()
        labels = self.model.predict(self.data)

        plt.scatter(self.data[:, 0], self.data[:, 1], c = labels, s = 15)

        if n == None:
            plt.show()
        else:
            plt.savefig('fig/dim_vbem_%d_%d' % (n, 4))


if __name__ == '__main__':
    gmm = VBEM(n_components = 4)
    gmm.train()
    gmm.show()