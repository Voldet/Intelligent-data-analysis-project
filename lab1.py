import pandas as pd
import os
from numpy import *
from matplotlib.pyplot import *
import numpy as np


class PcaMethod(object):
    '''Whole method principle:
           Main idea: project the original data set(n) to a new
             coordinate space containing k axis where k<<n'''
    def __init__(self):
        self.data = []
        self.normal_data = []
        self.project_data = []
        self.num_pricipal_com = 2
        self.eigen_va = []
        self.eigen_ve = []
        self.var = 0
        self.cum_sum = []

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
        self.normal_data = ans

    def eigval_pct(self, percentage):
        '''calculate how many principal components should we need'''
        arraySum=sum(self.eigen_va)
        tempSum=0
        num=0
        for i in self.eigen_va:
            tempSum+=i
            num+=1
            if tempSum>=arraySum*percentage:
                print(num)
                self.num_pricipal_com = num

    def best_eig(self):
        '''show the proportion of each principal component'''
        bar(range(len(self.eigen_va)), self.var_exp, width=1.0, bottom=0.0, alpha=0.5,
                label='individual eigenvalue proportion', color='coral')
        plot(range(len(self.eigen_va)), self.var_exp, c='#000000', label='change of eigenvalue')
        step(range(len(self.eigen_va)), self.cum_var_exp, where='post', label='cumulative eigenvalue proportion')
        ylabel('Explained eigenvalue ratio')
        xlabel('Principal components')
        legend(loc='best')
        show()

    def pca(self):
        '''do the pca:
                why we need to calculate the covariance matrix?
                    '''
        cov_mat = mat(cov(self.normal_data, rowvar=0, ddof=0))
        eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
        red_eig_val = argsort(eigenvalues)
        self.eigen_va = eigenvalues
        self.eigen_ve = eigenvectors
        tot = np.sum(eigenvalues)
        self.var_exp = [(i / tot) for i in sorted(eigenvalues, reverse=True)]
        self.cum_var_exp = np.cumsum(self.var_exp)
        red_eig_val = red_eig_val[:-(self.num_pricipal_com + 1):-1]
        red_eig_vects = eigenvectors[:, red_eig_val]
        show()
        self.project_data = red_eig_vects.T.dot(self.normal_data.T)

    def show_data(self):
        _array = self.project_data.getA()
        print(type(self.data[0,0]))
        print(1L == self.data[0,0])
        low_level_x, low_level_y = [], []
        medium_level_x, medium_level_y = [], []
        high_level_x, high_level_y = [], []
        very_high_level_x, very_high_level_y = [],[]
        for i in range(0,int(self.project_data.shape[1])-1):
            if self.data[i,0] == 1L:
                low_level_x.append(_array[0][i])
                low_level_y.append(_array[1][i])
            elif self.data[i,0] == 2L:
                medium_level_x.append(_array[0][i])
                medium_level_y.append(_array[1][i])
            elif self.data[i,0] == 3L:
                high_level_x.append(_array[0][i])
                high_level_y.append(_array[1][i])
            else:
                very_high_level_x.append(_array[0][i])
                very_high_level_y.append(_array[1][i])
        scatter(low_level_x, low_level_y, c='k', marker='x',s=100, alpha=0.5)
        scatter(medium_level_x, medium_level_y, c='c', marker='D',s=50, alpha=0.5)
        scatter(high_level_x, high_level_y, c='g', marker='*',s=100, alpha=0.5)
        scatter(very_high_level_x, very_high_level_y, c='m', marker='.', s=50, alpha=0.5)
        show()


if __name__ == '__main__':
        Main = PcaMethod()
        Main.read_data(file_name='Dataset.csv',
                       path=r'C:\Users\machenike\Desktop\IDA_Assignment\Dataset_folder\wine')
        Main.normalization()
        Main.eigval_pct(0.6)
        Main.pca()
        Main.best_eig()
        Main.show_data()