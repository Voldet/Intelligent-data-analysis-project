import pandas as pd
import os
from numpy import *
from matplotlib.pyplot import *
import numpy as np
from minisom import MiniSom


class PcaMethod(object):
    '''Whole method principle:
           Main idea: project the original data set(n) to a new
             coordinate space containing k axis where k<<n'''
    def __init__(self):
        self.label =np.zeros(178)
        self.data = []
        self.help_data = []
        self.normal_data = []
        self.project_data = []
        self.num_pricipal_com = 3
        self.eigen_va = []
        self.eigen_ve = []
        self.var = 0
        self.cum_sum = []
        self.low_level_x, self.low_level_y = [], []
        self.medium_level_x, self.medium_level_y = [], []
        self.high_level_x, self.high_level_y = [], []
        self.very_high_level_x, self.very_high_level_y = [], []
        self.low_level_z, self.medium_level_z, self.high_level_z, self.very_high_level_z = [], [], [], []

    def read_data(self, file_name, path):
        '''change the path and find the file name we need to do pca'''
        os.getcwd()
        os.chdir(path)
        self.help_data = array(pd.read_csv(file_name, encoding='utf-8'))
        data = delete(self.help_data, 0, axis=1)
        self.data = mat(data)

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
        # axvline(x = 3,linewidth=1.5, color = 'orangered',linestyle="--", label = ':"knee" point')
        axhline(y=self.cum_var_exp[1], linewidth=1.5, color='crimson', linestyle="--", label='0.6')
        ylabel('Explained eigenvalue ratio')
        xlabel('Principal components')
        legend(loc = 'lower right',prop={'size':8})

        show()

    def pca(self):
        '''do the pca:
                why we need to calculate the covariance matrix?
                    '''
        cov_mat = mat(cov(self.normal_data, rowvar=0, ddof=0))
        eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
        # print(eigenvalues)
        red_eig_val = argsort(eigenvalues)
        self.eigen_va = eigenvalues
        self.eigen_ve = eigenvectors
        tot = np.sum(eigenvalues)
        self.var_exp = [(i / tot) for i in sorted(eigenvalues, reverse=True)]
        self.cum_var_exp = np.cumsum(self.var_exp)
        red_eig_val = red_eig_val[:-(self.num_pricipal_com + 1):-1]
        red_eig_vects = eigenvectors[:, red_eig_val]
        # print(self.eigen_va)
        # print(red_eig_vects)
        show()
        self.project_data = red_eig_vects.T.dot(self.normal_data.T)

    def show_data(self):
        from mpl_toolkits.mplot3d import Axes3D
        ax = subplot(111,projection = '3d')
        _array = self.project_data.getA()
        malic_acid, Ash, alcalinity = self.data[:,0], self.data[:,1],self.data[:,2]
        print(malic_acid.shape)
        for i in range(0,int(self.project_data.shape[1])-1):
            if self.help_data[i,0] == 1L:
                self.low_level_x.append(_array[0][i])
                self.low_level_y.append(_array[1][i])
                self.low_level_z.append(_array[2][i])
                self.label[i] = 0
                # low_level_x.append(malic_acid[i])
                # low_level_y.append(Ash[i])
                # low_level_z.append(alcalinity[i])
            elif self.help_data[i,0] == 2L:
                self.medium_level_x.append(_array[0][i])
                self.medium_level_y.append(_array[1][i])
                self.medium_level_z.append(_array[2][i])
                self.label[i] = 1
                # medium_level_x.append(malic_acid[i])
                # medium_level_y.append(Ash[i])
                # medium_level_z.append(alcalinity[i])
            elif self.help_data[i,0] == 3L:
                self.high_level_x.append(_array[0][i])
                self.high_level_y.append(_array[1][i])
                self.high_level_z.append(_array[2][i])
                self.label[i] = 2
                # high_level_x.append(malic_acid[i])
                # high_level_y.append(Ash[i])
                # high_level_z.append(alcalinity[i])
            else:
                self.very_high_level_x.append(_array[0][i])
                self.very_high_level_y.append(_array[1][i])
                self.very_high_level_z.append(_array[2][i])

        ax.scatter(self.low_level_x, self.low_level_y, self.low_level_z, c='orangered', marker='x',s=10, alpha=0.5,label = 'Class 1')
        ax.scatter(self.medium_level_x, self.medium_level_y, self.medium_level_z, c='deepskyblue', marker='D',s=10, alpha=0.5,label = 'Class 2')
        ax.scatter(self.high_level_x, self.high_level_y, self.high_level_z, c='yellow', marker='*',s=10, alpha=0.5,label = 'Class 3')
        ax.scatter(self.very_high_level_x, self.very_high_level_y, self.very_high_level_z, c='m', marker='.', s=10, alpha=0.5)
        ax.set_xlabel('malic_acid')
        ax.set_ylabel('Ash')
        ax.set_zlabel('alcalinity')
        ax.legend(loc='upper left', prop={'size': 8})
        show()

    def som(self):
        import matplotlib.pyplot as plt
        som = MiniSom(13,13,3,learning_rate=0.5,sigma=2.5)

        self.project_data = self.project_data.T.getA()
        data = list(self.project_data)

        som.random_weights_init(data)

        som.train_random(data, 10000)
        '''SOM weight graph'''
        # for i in range(0, 13):
        #     plot(som._weights[i, :, 0]*3, som._weights[i, :, 1]*3, '.-', c='k')
        # for i in range(0, 13):
        #     plot(som._weights[:, i, 0]*3, som._weights[:, i, 1]*3, '.-', c='k')
        #
        # plt.scatter(self.low_level_x, self.low_level_y, c='orangered', marker='x', s=10, alpha=0.5,
        #            label='Class 1')
        # plt.scatter(self.medium_level_x, self.medium_level_y,  c='deepskyblue', marker='D', s=10,
        #            alpha=0.5, label='Class 2')
        # plt.scatter(self.high_level_x, self.high_level_y, c='yellow', marker='*', s=40,alpha=0.5,
        #            label='Class 3')
        # plt.show()
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #
        # for i in range(0, 13):
        #     ax.plot(som._weights[i, :, 0]*3, som._weights[i, :, 1]*3, som._weights[i, :, 2]*3, '.-', c='k')
        # for i in range(0, 13):
        #     ax.plot(som._weights[:, i, 0]*3, som._weights[:, i, 1]*3, som._weights[:, i, 2]*3, '.-', c='k')
        #
        # ax.scatter(self.low_level_x, self.low_level_y, self.low_level_z, c='orangered', marker='x', s=10, alpha=0.5,
        #            label='Class 1')
        # ax.scatter(self.medium_level_x, self.medium_level_y, self.medium_level_z, c='deepskyblue', marker='D', s=10,
        #            alpha=0.5, label='Class 2')
        # ax.scatter(self.high_level_x, self.high_level_y, self.high_level_z, c='yellow', marker='*', s=10, alpha=0.5,
        #            label='Class 3')
        # plt.show()
        '''SOM project graph'''

        from pylab import plot, axis, show, pcolor, colorbar, bone

        markers = ['x', 'D','*']
        colors = ['orangered', 'deepskyblue', 'yellow']
        print type(self.label[1])
        for cnt, xx in enumerate(data):
            w = som.winner(xx)  # getting the winner
            # palce a marker on the winning position for the sample xx
            plot(w[0] + .5, w[1] + .5, markers[int(self.label[cnt])], markerfacecolor='None',
                 markeredgecolor=colors[int(self.label[cnt])], markersize=12, markeredgewidth=2)
            print(self.label[cnt])
        axis([0, som._weights.shape[0], 0, som._weights.shape[1]])
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
        Main.som()