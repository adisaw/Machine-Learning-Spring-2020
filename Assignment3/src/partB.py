import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import math

#distance function
def calcdist(arr1,arr2):
    cos_sim=dot(arr1,arr2)
    return math.exp(-cos_sim)

#read csv
data=pd.read_csv('Processed.csv')
dist=np.zeros(shape=(data.shape[0],data.shape[0]))
val=np.zeros(shape=(data.shape[0],data.shape[1]-1))
val=data.iloc[:,1:]

#calculate distance between each vector
for i in range(val.shape[0]):
	for j in range(val.shape[0]):
		if i!=j :
			dist[i][j]=calcdist(val.iloc[i,:],val.iloc[j,:])
	print(i)

#making single element clusters
cluster=[]
for i in range (val.shape[0]):
    temp=[]
    temp.append(i)
    cluster.append(temp)
print(' ')
#Agglomerative clustering
for h in range(val.shape[0]-8):
    mindist=100000000
    cluster1=-1
    cluster2=-1
    for i in range(len(cluster)):
        for j in range(i+1,len(cluster)):
            for k in range(len(cluster[i])):
                for l in range(len(cluster[j])):
                    distm=dist[cluster[i][k]][cluster[j][l]]
                    if distm<mindist :
                        mindist=distm
                        cluster1=i
                        cluster2=j
    
    cluster[cluster1]=cluster[cluster1]+cluster[cluster2]
    cluster.remove(cluster[cluster2])
    print(h)
#writing to file
f=open("agglomerative.txt","w")

for i in range(len(cluster)):
	cluster[i].sort()
	for item in cluster[i]:
		f.write("%s," %item)
	f.write("\n")
#printing cluster
print(cluster)


    





