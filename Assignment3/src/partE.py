import numpy as np
import math

#calculate normalized mutual information
def calc_NMI(filename):
	#probability of a class
	probclass=np.zeros(8)
	probclass[0]=45/589
	probclass[1]=81/589
	probclass[2]=162/589
	probclass[3]=189/589
	probclass[4]=31/589
	probclass[5]=12/589
	probclass[6]=50/589
	probclass[7]=19/589


	H_Y=0.0
	for i in range(8):
		H_Y=H_Y-probclass[i]*math.log(probclass[i],2)

	#print(H_Y)
	#defining class boundaries
	Range=np.zeros(shape=(8,2))
	Range[0][0]=0;
	Range[0][1]=44;
	Range[1][0]=45;
	Range[1][1]=125;
	Range[2][0]=126;
	Range[2][1]=287;
	Range[3][0]=288;
	Range[3][1]=476;
	Range[4][0]=477;
	Range[4][1]=507;
	Range[5][0]=508;
	Range[5][1]=519;
	Range[6][0]=520;
	Range[6][1]=569;
	Range[7][0]=570;
	Range[7][1]=588;
	
	

	f=open(filename,"r")
	lines=f.readlines()
#initialize clusters
	cluster1=lines[0].split(',')
	cluster1.remove('\n')
	cluster1=np.asarray(cluster1,dtype=int)
	cluster2=lines[1].split(',')
	cluster2.remove('\n')
	cluster2=np.asarray(cluster2,dtype=int)
	cluster3=lines[2].split(',')
	cluster3.remove('\n')
	cluster3=np.asarray(cluster3,dtype=int)
	cluster4=lines[3].split(',')
	cluster4.remove('\n')
	cluster4=np.asarray(cluster4,dtype=int)
	cluster5=lines[4].split(',')
	cluster5.remove('\n')
	cluster5=np.asarray(cluster5,dtype=int)
	cluster6=lines[5].split(',')
	cluster6.remove('\n')
	cluster6=np.asarray(cluster6,dtype=int)
	cluster7=lines[6].split(',')
	cluster7.remove('\n')
	cluster7=np.asarray(cluster7,dtype=int)
	cluster8=lines[7].split(',')
	cluster8.remove('\n')
	cluster8=np.asarray(cluster8,dtype=int)
	cluster=[]
	cluster.append(cluster1)
	cluster.append(cluster2)
	cluster.append(cluster3)
	cluster.append(cluster4)
	cluster.append(cluster5)
	cluster.append(cluster6)
	cluster.append(cluster7)
	cluster.append(cluster8)

	
	probcluster=np.zeros(8)
	for i in range(8):
		probcluster[i]=len(cluster[i])/589
	H_C=0.0
	for i in range(8):
		H_C=H_C-probcluster[i]*(math.log(probcluster[i],2))

	#print(H_C)
	H_YC=0.0
	for i in range(8):
		temp=0.0
		for j in range(8):
			val=np.count_nonzero((cluster[i]>=Range[j][0])&(cluster[i]<=Range[j][1]))
			if val!=0:
				temp=temp+(val/len(cluster[i]))*math.log((val/len(cluster[i])),2)
		H_YC=H_YC-probcluster[i]*temp
		#print(H_YC)
	I_YC=H_Y-H_YC
	NMI=(2*I_YC)/(H_Y+H_C)
	print(filename,' NMI :',NMI)

#getting NMI for different files

calc_NMI("agglomerative.txt")
calc_NMI("kmeans.txt")
calc_NMI("agglomerative_reduced.txt")
calc_NMI("kmeans_reduced.txt")

