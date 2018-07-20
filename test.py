from sklearn.decomposition import PCA
import pandas as pd
import os
from numpy import *
from matplotlib.pyplot import *


def read_data(name, path):
    os.getcwd()
    os.chdir(path)
    data = pd.read_csv(name, encoding = 'utf-8')
    return data


def normalization(data):
    mean = data.mean(axis=0)
    # variance = data.var(axis=0)
    # sigma = np.sqrt(variance)
    sigma = std(data,axis =0)
    # print(sigma1)
    ans = ((data - mean) / sigma)
    # print(ans)
    return ans


if __name__ == '__main__':
    data = read_data('Dataset.csv', 'C:\Users\machenike\Desktop')
    data = mat(normalization(data))
    pca = PCA(n_components=2)
    pca.fit(data)
    PCA(copy=True, n_components=2,whiten =False)
    newMat = pca.transform(data)
    print(newMat.shape)
    scatter(data[:,0],data[:,1])
    show()