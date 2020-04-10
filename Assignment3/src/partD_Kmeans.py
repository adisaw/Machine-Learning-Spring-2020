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
data = np.genfromtxt('Processed_Reduced.csv', dtype=np.float64, delimiter=',')

dist=np.zeros(shape=(data.shape[0],data.shape[0]))

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




#centroid array
centroids=np.zeros(shape=(8,data.shape[1]))
for i in range(8):
	centroids[i,:]=data[cluster[i][0],:]

#perform k means clustering

for i in range(100):
    for j in range(data.shape[0]):
        mindist=10000000
        clust=-1
        for k in range(8):
            dist=calcdist(data[j,:],centroids[k,:])
            if dist<mindist:
                mindist=dist
                clust=k
        for k in range(8):
            if j in cluster[k]:
                cluster[k].remove(j)
        cluster[clust].append(j)
        
    
    for l in range(8):
        temp=np.zeros(data.shape[1])
        for m in range(len(cluster[l])):
            temp=np.add(temp,data[cluster[l][m],:])
        centroids[l,:]=temp/len(cluster[l])
    print(i)
    

#write data
f=open("kmeans_reduced.txt","w")

for i in range(len(cluster)):
	cluster[i].sort()

cluster.sort()

for i in range(len(cluster)):
	for item in cluster[i]:
		f.write("%s," %item)
	f.write("\n")
#print
print(cluster)
