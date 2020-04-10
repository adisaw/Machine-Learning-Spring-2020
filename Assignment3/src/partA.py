import numpy as np
import pandas as pd
import math
from sklearn import preprocessing
#Reading csv
data1=pd.read_csv('AllBooks_baseline_DTM_Labelled.csv')
#dropping zero row
data1=data1.drop(data1.index[13]);

#Removing Ch1 etc
for i in range(589):
	word=data1.iloc[i,0].split('_')
	data1.iloc[i,0]=word[0];


idfarr=np.zeros(data1.shape[1]-1)

tfmatrix=np.zeros(shape=(data1.shape[0],data1.shape[1]-1))

tfmatrix=data1.iloc[:,1:];

p=np.count_nonzero(tfmatrix,axis=0)
#idf vector
for i in range (idfarr.size):
	idfarr[i]=math.log((1+tfmatrix.shape[0])/(1+p[i]));

#diagonalizing idf vector
diagidf=np.diag(idfarr)

tf_idf=np.zeros(shape=(data1.shape[0],data1.shape[1]-1))
#getting tf-idf matrix
tf_idf=np.matmul(tfmatrix,diagidf)
#L2 normalization
tf_idf_l2 = preprocessing.normalize(tf_idf, norm='l2')

data1.iloc[:,1:]=tf_idf_l2
#storing data to csv
data1.to_csv('Processed.csv',index=False)