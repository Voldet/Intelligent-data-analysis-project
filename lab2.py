import pandas as pd
import os
from numpy import *
from matplotlib.pyplot import *
import numpy as np
from sklearn.decomposition import PCA
from minisom import  *


class SomMethod(object):
    def __init__(self):
        self.data = []
        self.project_data = []
        self.normalized_data = []

    def read_data(self, file_name, path):
        '''change the path and find the file name we need to do pca'''
        os.getcwd()
        os.chdir(path)
        self.data = mat(pd.read_csv(file_name, encoding='utf-8'))

    def normalization(self):
        '''answer is the normalized data.
           Why?
            Because the unit of the data is different,
            this method can make different characters have the same scale.
            Only in this way, the comparision can be valid'''
        _mean = self.data.mean(axis=0)
        sigma = std(self.data, axis=0)
        '''get mean value of each column and
            get sigma of each column'''
        ans = ((self.data - _mean) / sigma)
        self.normalized_data = ans

    def pca(self):
        pca = PCA(n_components=3)
        pca.fit(self.normalized_data)
        PCA(copy=True, n_components=3, whiten=False)
        self.project_data =mat(pca.transform(self.normalized_data))
        print(self.project_data.shape)

    def show_data(self):
        _array = self.project_data.getA()
        print(_array.shape)
        print(type(self.data[0,0]))
        print(1L == self.data[0,0])
        low_level_x, low_level_y = [], []
        medium_level_x, medium_level_y = [], []
        high_level_x, high_level_y = [], []
        very_high_level_x, very_high_level_y = [],[]
        for i in range(0,int(self.project_data.shape[0])-1):
            if self.data[i,0] == 1L:
                low_level_x.append(_array[i][0])
                low_level_y.append(_array[i][1])
            elif self.data[i,0] == 2L:
                medium_level_x.append(_array[i][0])
                medium_level_y.append(_array[i][1])
            elif self.data[i,0] == 3L:
                high_level_x.append(_array[i][0])
                high_level_y.append(_array[i][1])
            else:
                very_high_level_x.append(_array[i][0])
                very_high_level_y.append(_array[i][1])
        scatter(low_level_x, low_level_y, c='k', marker='x',s=100, alpha=0.5)
        scatter(medium_level_x, medium_level_y, c='c', marker='D',s=50, alpha=0.5)
        scatter(high_level_x, high_level_y, c='g', marker='*',s=100, alpha=0.5)
        scatter(very_high_level_x, very_high_level_y, c='m', marker='.', s=50, alpha=0.5)
        show()

    def som(self):
        I = zeros(178)
        som = MiniSom(20,20,3,learning_rate=0.5,sigma=0.1)
        print(self.project_data.shape)
        data = list(self.project_data)
        som.train_random(data, 400)
        pcolor(som.distance_map().T)
        marker = ['x','D','*']
        color = ['orangered','deepskyblue','yellow']
        for i, x in enumerate(self.project_data):
            w = som.winner(x)
            plot(w[0] + .5, w[1] + .5, marker[I[i].astype(np.int64)], markerfacecolor='None',
                 markeredgecolor=color[I[i].astype(np.int64)], markersize=5, markeredgewidth=2)

        axis([0, 20, 0, 20])
        show()


if __name__ == '__main__':
    Main = SomMethod()
    Main.read_data(file_name='Dataset.csv',
                   path=r'C:\Users\machenike\Desktop\IDA_Assignment\Dataset_folder\wine')
    Main.normalization()
    Main.pca()
    Main.show_data()
    Main.som()