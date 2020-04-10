import numpy as np 



#loading data from csv file
traindata = np.genfromtxt('winequality-red.csv', dtype=np.float32, delimiter=';', skip_header=1)
 
mat=np.asmatrix(traindata)

#min max scaling
for i in range(11):
	temp=mat[:,i]
	mat[:,i]=(temp-temp.min())/(temp.max()-temp.min())

#setting output class
for i in range(mat.shape[0]):
	if mat[i,11]<=6 :
		mat[i,11]=0
	else :
		mat[i,11]=1


#saving csv for logistic regression
np.savetxt("logisticregression.csv",mat,delimiter=',')
train = np.genfromtxt('winequality-red.csv', dtype=np.float32, delimiter=';', skip_header=1)

mat2=np.asmatrix(train)

#setting output class for decision tree
for i in range(mat2.shape[0]):
	if mat2[i,11]<5 :
		mat2[i,11]=0
	elif mat2[i,11]==5 or mat2[i,11]==6 :
		mat2[i,11]=1
	else :
		mat2[i,11]=2
#zcsore normalization
for i in range(11):
	temp=mat2[:,i]
	mat2[:,i]=(temp-temp.mean())/(temp.std())


#print(mat2)

#assigning bins
for i in range(11):
	maxval=mat2[:,i].max()
	minval=mat2[:,i].min()
	diff=maxval-minval
	fq=minval+0.25*diff
	mq=minval+0.5*diff
	lq=minval+0.75*diff
	for j in range(mat2.shape[0]):
		if mat2[j,i]>=minval and mat2[j,i]<fq:
			mat2[j,i]=0
		elif mat2[j,i]>=fq and mat2[j,i]<mq:
			mat2[j,i]=1
		elif mat2[j,i]>=mq and mat2[j,i]<lq:
			mat2[j,i]=2
		elif mat2[j,i]>=lq and mat2[j,i]<=maxval:
			mat2[j,i]=3


#saving csv for decision tree
np.savetxt("decisiontree.csv",mat2,delimiter=',')


