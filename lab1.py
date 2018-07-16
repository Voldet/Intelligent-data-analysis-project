import pandas as pd
import os
from numpy import *
from matplotlib.pyplot import *
import numpy as np


class PcaMethod(object):
    def __init__(self):
        self.data = []
        self.normal_data = []
        self.project_data = []

    def read_data(self, file_name, path):
        os.getcwd()
        os.chdir(path)
        self.data = mat(pd.read_csv(file_name, encoding='utf-8'))

    def normalization(self):
        mean = self.data.mean(axis=0)
        sigma = std(self.data,axis=0)
        ans = ((self.data - mean) / sigma)
        self.normal_data = ans

    def eigval_pct(self,eig_vals, percentage):
        arraySum=sum(eig_vals)
        tempSum=0
        num=0
        for i in eig_vals:
            tempSum+=i
            num+=1
            if tempSum>=arraySum*percentage:
                return num

    def pca(self, num):
        cov_mat = mat(cov(self.normal_data, rowvar=0, ddof=0))
        eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
        red_eig_val = argsort(eigenvalues)
        # print(eigenvalues)
        # print(red_eig_val)
        red_eig_val = red_eig_val[:-(num+1):-1]
        red_eig_vects = eigenvectors[:, red_eig_val]
        # print(red_eig_val)
        # print(red_eig_vects)
        show()
        self.project_data = red_eig_vects.T.dot(self.normal_data.T)

    def show_data(self):
        scatter(self.project_data[0], self.project_data[1])
        show()

    def make_color(m):
        labels = []
        index =0
        for i in range(m.shape(1)):
            array = m[:,i]
            labels.append(array[:])
            index+=1
        return labels

if __name__ == '__main__':
        Main = PcaMethod()
        Main.read_data(file_name='Dataset.csv', path='C:\Users\machenike\Desktop')
        Main.normalization()
        Main.pca(2)
        Main.show_data()