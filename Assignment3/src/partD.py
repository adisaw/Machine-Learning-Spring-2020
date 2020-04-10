import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import random
from sklearn.decomposition import PCA
from sklearn import preprocessing
#read csv
data=pd.read_csv('Processed.csv')
dist=np.zeros(shape=(data.shape[0],data.shape[0]))
val=np.zeros(shape=(data.shape[0],data.shape[1]-2))
val=data.iloc[:,2:]

pca=PCA(n_components=100)
#reducing components
principal=pca.fit_transform(val)

#normalize data
principal = preprocessing.normalize(principal, norm='l2')
#saving csv
np.savetxt("Processed_Reduced.csv",principal,delimiter=',')