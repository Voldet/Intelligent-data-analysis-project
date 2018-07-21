import pandas as pd
import os
from numpy import *
from matplotlib.pyplot import *
import numpy as np
from SOM import Som


class PcaMethod(object):
    '''Whole method principle:
           Main idea: project the original data set(n) to a new
             coordinate space containing k axis where k<<n'''
    def __init__(self):
        self.label =np.zeros(1599)
        self.data = []
        self.help_data = []
        self.normal_data = []
        self.project_data = []
        self.num_pricipal_com = 5
        self.eigen_va = []
        self.eigen_ve = []
        self.var = 0
        self.cum_sum = []
        self.low_level_x, self.low_level_y = [], []
        self.medium_level_x, self.medium_level_y = [], []
        self.high_level_x, self.high_level_y = [], []
        self.very_high_level_x, self.very_high_level_y = [], []
        self.low_level_z, self.medium_level_z, self.high_level_z, self.very_high_level_z = [], [], [], []
        self._low_level_x, self._low_level_y = [], []
        self._medium_level_x, self._medium_level_y = [], []
        self._high_level_x, self._high_level_y = [], []
        self._very_high_level_x, self._very_high_level_y = [], []
        self._low_level_z, self._medium_level_z, self._high_level_z, self._very_high_level_z = [], [], [], []
        self.huge_level_x, self.huge_level_y, self.huge_level_z,= [], [], []
        self.Dfram_data = []
        self.attribu = ['fixed acidity',
                        "volatile acidity",
                        "citric acid" ,
                        "residual sugar" ,
                        "chlorides" ,
                        "free sulfur dioxide",
                        "density"  ,
                        "pH"   ,
                        "sulphates"  ,
                        "alcohol"  ,
                        "quality" ]


    def read_data(self, file_name, path):
        '''change the path and find the file name we need to do pca'''
        os.getcwd()
        os.chdir(path)
        self.help_data = array(pd.read_csv(file_name, encoding='utf-8'))
        data = delete(self.help_data, 11, axis=1)
        self.data = mat(data)
        # print(self.data)
        self.Dfram_data =pd.read_csv(file_name, encoding='utf-8')
        # print type(self.Dfram_data)

    def d_normal(self):
        '''find organization of the principal component'''
        data_mean = self.Dfram_data.mean(0)
        centered_data = self.Dfram_data - data_mean
        centered_data.head()
        data_std = centered_data.std(0)
        std_data = centered_data / data_std
        final_data = std_data.transpose()
        self.data = final_data.drop(final_data.index[11])
        data_t = self.data.transpose()
        mean = []
        cov_mat = mat(cov(self.normal_data, rowvar=0, ddof=0))
        eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
        small = eigenvectors[:,1]
        print(small.shape)
        to_print = []
        for i in range(len(small)):
            t = float(small[i])
            x = [self.attribu[i],t]
            to_print.append(x)
        to_print.sort(key = lambda  to_print: abs(to_print[1]), reverse= True )
        print(to_print)
        print eigenvalues

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

    def label1(self):
        y = zeros(5)
        alcohol = mat(self.help_data)[:,11]
        # print alcohol[0]
        for i in range(len(alcohol)):
            if alcohol[i] <= 4:
                y[0]+=1
            elif alcohol[i] == 5:
                y[1] += 1
            elif alcohol[i] == 6:
                y[2] += 1
            elif alcohol[i] == 7:
                y[3] += 1
            else:
                y[4] += 1
        x = np.arange(5) + 1
        color = ['r','g','b','y','m']
        print(y[0],y[1],y[2],y[3],y[3])
        bar(x, y, color = color,width=.9)
        title("Histogram \"Wine Quality Class(Strategy 2)\"")
        xlabel("Class")
        ylabel("Number of data")
        show()

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
        axvline(x = 2,linewidth=1.5, color = 'orangered',linestyle="--", label = ':"knee" point')
        # axhline(y=self.cum_var_exp[2], linewidth=1.5, color='crimson', linestyle="--", label='3 cumulate components')
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
        # print(red_eig_vects.shape)
        show()
        self.project_data = red_eig_vects.T.dot(self.normal_data.T)

    def show_data(self):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        _array = self.project_data.getA()
        # _array = np.apply_along_axis(lambda x: x / np.linalg.norm(x), 1, _array)
        print(_array)
        # malic_acid, Ash, alcalinity = self.data[:,0], self.data[:,1],self.data[:,2]
        # print(malic_acid.shape)
        # print self.project_data.shape[1]
        for i in range(0,int(self.project_data.shape[1])-1):
            if self.help_data[i,6] < 100:
                self.low_level_x.append(_array[0][i])
                self.low_level_y.append(_array[1][i])
                # self.low_level_z.append(_array[2][i])
                self.label[i] = int(0)
                # print self.help_data[i,6] < 100

                # low_level_x.append(malic_acid[i])
                # low_level_y.append(Ash[i])
                # low_level_z.append(alcalinity[i])
            # elif self.help_data[i,11] == 2L:
            #     self.medium_level_x.append(_array[0][i])
            #     self.medium_level_y.append(_array[1][i])
            #     self.medium_level_z.append(_array[2][i])
            #     self.label[i] = 1
            #     # medium_level_x.append(malic_acid[i])
            #     # medium_level_y.append(Ash[i])
            #     # medium_level_z.append(alcalinity[i])
            # elif self.help_data[i,11] == 3L:
            #     self.high_level_x.append(_array[0][i])
            #     self.high_level_y.append(_array[1][i])
            #     self.high_level_z.append(_array[2][i])
            #     self.label[i] = 2
            #     # high_level_x.append(malic_acid[i])
            #     # high_level_y.append(Ash[i])
            #     # high_level_z.append(alcalinity[i])
            else:
                self.very_high_level_x.append(_array[0][i])
                self.very_high_level_y.append(_array[1][i])
                self.label[i] = int(1)

                # self.very_high_level_z.append(_array[2][i])
        print 'finish label'
        # ax.scatter(self.low_level_x, self.low_level_y,self.low_level_z, c='orangered', marker='x',s=20, alpha=0.5,label = 'Class 1')
        # # ax.scatter(self.medium_level_x, self.medium_level_y, self.medium_level_z, c='deepskyblue', marker='D',s=10, alpha=0.5,label = 'Class 2')
        # # ax.scatter(self.high_level_x, self.high_level_y, self.high_level_z, c='yellow', marker='*',s=10, alpha=0.5,label = 'Class 3')
        # ax.scatter(self.very_high_level_x, self.very_high_level_y,self.very_high_level_z, c='m', marker='.', s=20, alpha=0.5,label = 'Class 2')
        # xlabel('1st Principal Component')
        # ylabel('2st Principal Component')
        # title("3D projection of PCA ")
        # legend(loc='upper left', prop={'size': 8})
        # show()
        # print type(self.data)

    def som(self):
        import matplotlib.pyplot as plt
        from matplotlib import pyplot
        som = Som(7,7,5,learning_rate=3.5,sigma=1.9)
        self.project_data = self.project_data.T.getA()
        data = list(self.project_data)
        data = np.apply_along_axis(lambda x: x / np.linalg.norm(x), 1, data)
        # data = self.data.getA()

        som.random_weights_init(data)

        error = som.train_random(data, 5)
        # x = np.arange(5)+ 1
        # plt.bar(x, error,width=0.5)
        # plt.xlabel('Iteration Time')
        # plt.ylabel('Quantization Error')
        # plt.show()
        label = np.zeros(1599)
        for i in range(0,int(self.project_data.shape[0])-1):
            if self.help_data[i,11] <=4:
                label[i] = int(0)
            elif self.help_data[i,11] == 5:
                label[i] = int(1)
            elif self.help_data[i,11] == 6:
                label[i] = int(2)
            elif self.help_data[i,11] == 7:
                label[i] = int(3)
            else:
                label[i] = int(4)

        '''SOM weight graph'''
        for i in range(0, 7):
            plt.plot(som._weights[i, :, 0]*4, som._weights[i, :, 1]*4, '.-', c='k')
        for i in range(0, 7):
            plt.plot(som._weights[:, i, 0]*4, som._weights[:, i, 1]*4, '.-', c='k')

        plt.scatter(self.low_level_x, self.low_level_y, c='orangered', marker='x', s=40, alpha=0.5,
                   label='Class 1')
        # plt.scatter(self.medium_level_x, self.medium_level_y,  c='deepskyblue', marker='D', s=10,
        #            alpha=0.5, label='Class 2')
        plt.scatter(self.very_high_level_x, self.very_high_level_y, c='yellow', marker='*', s=40,alpha=0.5,
                   label='Class 2')
        plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #
        # for i in range(0, 40):
        #     ax.plot(som._weights[i, :, 0]*3, som._weights[i, :, 1]*3, som._weights[i, :, 2]*3, '.-', c='k')
        # for i in range(0, 40):
        #     ax.plot(som._weights[:, i, 0]*3, som._weights[:, i, 1]*3, som._weights[:, i, 2]*3, '.-', c='k')
        #
        # ax.scatter(self.low_level_x, self.low_level_y, self.low_level_z, c='orangered', marker='x', s=10, alpha=0.5,
        #            label='Class 1')
        # # ax.scatter(self.medium_level_x, self.medium_level_y, self.medium_level_z, c='deepskyblue', marker='D', s=10,
        # #            alpha=0.5, label='Class 2')
        # ax.scatter(self.very_high_level_x, self.very_high_level_y, self.very_high_level_y, c='yellow', marker='*', s=10, alpha=0.5,
        #            label='Class 2')
        # plt.show()
        '''SOM project graph'''

        from pylab import plot, axis, show, pcolor, colorbar, bone

        # markers = ['x', 'D','*']
        # colors = ['orangered', 'deepskyblue', 'yellow']
        # # print self.label[100]
        # for cnt, xx in enumerate(data):
        #     i = int(self.label[cnt])
        #     print i
        #     print type(colors[i])
        #     w = som.winner(xx)  # getting the winner
        #     # palce a marker on the winning position for the sample xx
        #     plot(w[0] + .5, w[1] + .5, markers[i], markerfacecolor='None',
        #          markeredgecolor=colors[i], markersize=12, markeredgewidth=2)
        #     # print(self.label[cnt])
        # axis([0, som._weights.shape[0], 0, som._weights.shape[1]])
        # show()
        markers = ['o','s','D','x','.']
        colors = ['r','g','b','y','m']
        for cnt,xx in enumerate(data):
         w = som.winner(xx) # getting the winner
         # palce a marker on the winning position for the sample xx
         plot(w[0]+.5+np.random.randn()/20,w[1]+.5+np.random.randn()/20,markers[int(label[cnt])],markerfacecolor='None',
           markeredgecolor=colors[int(label[cnt])],markersize=10,markeredgewidth=2)
        axis([0,som._weights.shape[0],0,som._weights.shape[1]])
        show() # show the figure

    def cluster(self):
        from mpl_toolkits.mplot3d import Axes3D

        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans
        quan_error = np.zeros(6)
        quan_error_ = np.zeros(6)
        trans = self.data.getA()
        print(trans[0])
        kmeans = KMeans(n_clusters=5, random_state=0).fit(trans)
        labels = kmeans.labels_
        kmeans.predict([trans[0], trans[1]])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for j in range(1599):
            quan_error[0] += np.sum(np.abs(trans[j, :] - kmeans.cluster_centers_[labels[j]]))

        self.project_data = self.project_data.T.getA()
        print quan_error
        for j in range(1599):
            if labels[j] == 0:
                self._low_level_x.append(self.project_data[j, 0])
                self._low_level_y.append(self.project_data[j, 1])
                self._low_level_z.append(self.project_data[j, 2])

            elif labels[j] == 1:
                self._medium_level_x.append(self.project_data[j, 0])
                self._medium_level_y.append(self.project_data[j, 1])
                self._medium_level_z.append(self.project_data[j, 2])
            elif labels[j] == 2:
                self._high_level_x.append(self.project_data[j, 0])
                self._high_level_y.append(self.project_data[j, 1])
                self._high_level_z.append(self.project_data[j, 2])
            elif labels[j] == 3:
                self._very_high_level_x.append(self.project_data[j, 0])
                self._very_high_level_y.append(self.project_data[j, 1])
                self._very_high_level_z.append(self.project_data[j, 2])
            else:
                self.huge_level_x.append(self.project_data[j, 0])
                self.huge_level_y.append(self.project_data[j, 1])
                self.huge_level_z.append(self.project_data[j, 2])
        # plt.legend(loc = 'lower left',prop={'size':8})
        ax.scatter(self._low_level_x, self._low_level_y, self._low_level_z, color='mediumorchid', s=7,
                   label='Class 1')
        ax.scatter(self._medium_level_x, self._medium_level_y, self._medium_level_z, color='darkorange', s=7,
                   label='Class 2')
        ax.scatter(self._high_level_x, self._high_level_y, self._high_level_z, color='deepskyblue', s=7,
                   label='Class 3')
        ax.scatter(self._very_high_level_x, self._very_high_level_y, self._very_high_level_z, color='green', s=7,
                   label='Class 4')
        ax.scatter(self.huge_level_x, self.huge_level_y, self.huge_level_z, color='yellow', s=7,
                   label='Class 5')

        ax.set_title("3D projection of the 5-means clustering results")
        ax.set_xlabel("1st Principle Component")
        ax.set_ylabel("2nd Principle Component")
        ax.set_zlabel("3rd Principle Component")
        ax.legend(loc='lower left', prop={'size': 8})
        plt.show()
        err_pca = [7587, 6634, 6081, 5813,5604]
        err = [19894, 14273, 11861, 10917, 10178]
        plot(arange(5)+1, err_pca)
        plt.xlabel("Number of Clusters")
        plt.ylabel("After Doing PCA Clustering Quantization Error")
        plt.axvline(x=3, linewidth=1.5, color='orangered', linestyle="--", label=':"knee" point')
        plt.legend(loc='upper right', prop={'size': 8})
        show()


if __name__ == '__main__':
        Main = PcaMethod()
        Main.read_data(file_name='Dataset.csv',
                       path=r'F:\data_analysis\venv1\Dataset_folder\wine_quality')
        Main.normalization()

        # Main.label1()
        # Main.eigval_pct(0.6)
        # Main.pca()
        # Main.best_eig()
        # Main.show_data()
        # Main.som()
        # Main.cluster()
        Main.d_normal()