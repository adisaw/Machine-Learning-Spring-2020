import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import random
import math

#distance function
def calcdist(arr1,arr2):
    cos_sim=dot(arr1,arr2)/norm(arr2)
    return math.exp(-cos_sim)

#read csv
data=pd.read_csv('Processed.csv')
dist=np.zeros(shape=(data.shape[0],data.shape[0]))
val=np.zeros(shape=(data.shape[0],data.shape[1]-1))
val=data.iloc[:,1:]

#getting random centers
tempcluster=[]
while 1:
	num=random.randint(0,data.shape[0]-1)
	if num not in tempcluster:
		tempcluster.append(num)
	if len(tempcluster)==8:
		break

cluster=[]
for i in range(8):
	temp=[]
	temp.append(tempcluster[i])
	cluster.append(temp)


#initializing centroid array
centroids=np.zeros(shape=(8,data.shape[1]-1))
for i in range(8):
	centroids[i,:]=val.iloc[cluster[i][0],:]


#kmeans computation
for i in range(100):
    for j in range(data.shape[0]):
        mindist=10000000
        clust=-1
        for k in range(8):
            dist=calcdist(val.iloc[j,:],centroids[k,:])
            if dist<mindist:
                mindist=dist
                clust=k
        for k in range(8):
            if j in cluster[k]:
                cluster[k].remove(j)
        cluster[clust].append(j)
        
    
    for l in range(8):
        temp=np.zeros(data.shape[1]-1)
        for m in range(len(cluster[l])):
            temp=np.add(temp,val.iloc[cluster[l][m],:])
        centroids[l,:]=temp/len(cluster[l])
    print(i)
    

#write to file
f=open("kmeans.txt","w")

for i in range(len(cluster)):
	cluster[i].sort()

cluster.sort()

for i in range(len(cluster)):
	for item in cluster[i]:
		f.write("%s," %item)
	f.write("\n")
#print cluster
print(cluster)

