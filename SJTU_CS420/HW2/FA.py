from sklearn.decomposition import FactorAnalysis
from dataset import Dataset
import matplotlib.pyplot as plt
from copy import deepcopy
import math

class FA(object):
    def __init__(self, n_components = 10, Data = None):
        '''
        :param n_components: dimensionality of y
        :param Data: dataset
        '''
        self.model = FactorAnalysis(n_components = n_components)

        self.n_components = n_components
        if Data == None:
            self.dataset = Dataset()
            self.dataset.generate()
        else:
            self.dataset = Data
        self.data = self.dataset.data
        self.aic = []
        self.bic = []
        self.aic_b = None
        self.colors = ['red', 'blue']

    def train(self):
        self.model.fit(self.data)

    def aic_select(self):
        '''
        using AIC to select model
        :return: None
        '''
        self.aic_b = True
        low = 99999
        for n in range(1, self.n_components + 1):
            fa = FactorAnalysis(n_components = n)
            fa.fit(self.data)
            dm = 2*self.data.shape[1]*(n+1)-n*(n+1)
            #dm = 2*self.data.shape[1]*n
            aic = -2*fa.score(self.data)*self.data.shape[0] + dm
            self.aic.append(aic)
            if self.aic[-1] < low:
                low = self.aic[-1]
                self.res_n = n
                self.model = deepcopy(fa)
        # print('------aic-------\n', self.aic)
        # self.res_n = self.aic.index(low) + 1
        print('selected components:', self.res_n, '\n')

    def bic_select(self):
        '''
        using BIC to select model
        :return: None
        '''
        self.aic_b = False
        low = 99999
        for n in range(1, self.n_components + 1):
            fa = FactorAnalysis(n_components=n)
            fa.fit(self.data)
            dm = 2*self.data.shape[1]*(n+1)-n*(n+1)
            #dm = 2*self.data.shape[1] * n
            bic = -2 * fa.score(self.data) * self.data.shape[0] + math.log(self.data.shape[0])*dm/2
            self.bic.append(bic)
            if self.bic[-1] < low:
                low = self.bic[-1]
                self.res_n = n
                self.model = deepcopy(fa)
        # print('------bic-------\n', self.bic)
        #self.res_n = self.bic.index(low) + 1
        print('selected components:', self.res_n, '\n')

    def show(self, n = None):
        '''
        show the result of trained model
        :param n: just used for save files
        :return: None
        '''
        # plt.figure()
        # labels = self.model.transform(self.data)
        # plt.scatter(self.data[:, 0], self.data[:, 1], c = self.colors[0], s = 15)

        return self.res_n


if __name__ == '__main__':
    fa = FA(n_components = 5)
    fa.aic_select()
    fa.bic_select()