import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import math

#distance funtion
def calcdist(arr1,arr2):
    cos_sim=dot(arr1,arr2)
    return math.exp(-cos_sim)

#read csv
data = np.genfromtxt('Processed_Reduced.csv', dtype=np.float64, delimiter=',')
dist=np.zeros(shape=(data.shape[0],data.shape[0]))

#distance matrix
for i in range(data.shape[0]):
	for j in range(data.shape[0]):
		if i!=j :
			dist[i][j]=calcdist(data[i,:],data[j,:])
	print(i)

cluster=[]
for i in range (data.shape[0]):
    temp=[]
    temp.append(i)
    cluster.append(temp)
print(' ')
#Agglomerative clustering
for h in range(data.shape[0]-8):
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
#writing data
f=open("agglomerative_reduced.txt","w")

for i in range(len(cluster)):
	cluster[i].sort()
	for item in cluster[i]:
		f.write("%s," %item)
	f.write("\n")
#print cluster
print(cluster)


    





