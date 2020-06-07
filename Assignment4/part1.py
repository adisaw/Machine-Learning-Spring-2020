import numpy as np
from random import random
from numpy.random import rand
import matplotlib.pyplot as plt

#Preprocess function
def Preprocess():
	#loads data.txt file
	data=np.genfromtxt('data.txt', dtype=np.float64, delimiter='	')
	mat=np.asmatrix(data)
	
	#Z-score normalization
	for i in range(7):
		temp=mat[:,i]
		mat[:,i]=(temp-temp.mean())/(temp.std());

	np.random.shuffle(data)
	traindata=mat[0:168,:]
	testdata=mat[168:210,:]
	#Saving train and test datasets
	np.savetxt("train.csv",traindata,delimiter=',')
	np.savetxt("test.csv",testdata,delimiter=',')

#Load the train and test set
def DataLoader():
	global train
	global test
	train=np.genfromtxt('train.csv',dtype=np.float64,delimiter=',')
	test=np.genfromtxt('test.csv',dtype=np.float64,delimiter=',')


#Initialiaze the weights
def weightinitializer(part):
	#Specification 1A
	if part==1:
		global wp1_1
		global wp1_2
		global bias1_1
		global bias1_2
		wp1_1=-1+2*rand(7,32)
		wp1_2=-1+2*rand(32,3)
		bias1_1=-1+2*rand(32)
		bias1_2=-1+2*rand(3)
	#Specification 1B
	if part==2:
		global wp2_1
		global wp2_2
		global wp2_3
		global bias2_1
		global bias2_2
		global bias2_3
		wp2_1=-1+2*rand(7,64)
		wp2_2=-1+2*rand(64,32)
		wp2_3=-1+2*rand(32,3)
		bias2_1=-1+2*rand(64)
		bias2_2=-1+2*rand(32)
		bias2_3=-1+2*rand(3)
			

#for forward pass of neural network
def forward(minibatch,part):
	if part==1:
		global wp1_1
		global wp1_2
		global bias1_1
		global bias1_2
		s1=np.matmul(minibatch,wp1_1)
		s1=s1+bias1_1
		x1=1.0/(1.0+np.exp(-s1))
		s2=np.matmul(x1,wp1_2)
		s2=s2+bias1_2
		x2t=np.exp(s2)
		x2sum=np.sum(x2t,axis=1)
		x2=x2t
		for i in range(minibatch.shape[0]):
			x2[i,:]=x2[i,:]/x2sum[i]
		return x1,x2
	if part==2:
		global wp2_1
		global wp2_2
		global wp2_3
		global bias2_1
		global bias2_2
		global bias2_3
		s1=np.matmul(minibatch,wp2_1)
		s1=s1+bias2_1
		x1=np.where(s1<0,0,s1)
		s2=np.matmul(x1,wp2_2)
		s2=s2+bias2_2
		x2=np.where(s2<0,0,s2)
		s3=np.matmul(x2,wp2_3)
		s3=s3+bias2_3
		x3t=np.exp(s3)
		x3sum=np.sum(x3t,axis=1)
		x3=x3t
		for i in range(minibatch.shape[0]):
			x3[i,:]=x3[i,:]/x3sum[i]
		return x1,x2,x3



#Backpropagation and updating weights
def backward(minibatch,onehotenc,x1,x2,x3,part):
	if part==1:
		global wp1_1
		global wp1_2
		global bias1_1
		global bias1_2
		
		delta1=x2-onehotenc
		tempones=np.ones(shape=(x1.shape[0],x1.shape[1]))
		tempones=tempones-x1
		ans=np.multiply(tempones,x1)
		we_del_pro=np.matmul(wp1_2,delta1.transpose())
		delta2=np.multiply(ans.transpose(),we_del_pro)
		w2temp=np.zeros(shape=(32,3))
		x1ver=np.zeros(shape=(32,1))
		d1hor=np.zeros(shape=(1,3))
		b2temp=np.zeros(3)
		for i  in range(minibatch.shape[0]):
			x1ver[:,0]=x1[i,:].transpose()
			d1hor[0,:]=delta1[i,:]
			w2temp=w2temp+wp1_2-0.01*np.matmul(x1ver,d1hor)
			b2temp=b2temp+bias1_2-0.01*delta1[i,:]
			
		wp1_2=w2temp/minibatch.shape[0]
		bias1_2=b2temp/minibatch.shape[0]
		minitemp=minibatch.transpose()
		w1temp=np.zeros(shape=(7,32))
		b1temp=np.zeros(32)
		miniver=np.zeros(shape=(7,1))
		d2hor=np.zeros(shape=(1,32))
		delta2=delta2.transpose()
		for i in range(minibatch.shape[0]):
			miniver[:,0]=minitemp[:,i]
			d2hor[0,:]=delta2[i,:]
			w1temp=w1temp+wp1_1-0.01*np.matmul(miniver,d2hor)
			b1temp=b1temp+bias1_1-0.01*delta2[i,:]
		wp1_1=w1temp/minibatch.shape[0]
		bias1_1=b1temp/minibatch.shape[0]

	if part==2:
		global wp2_1
		global wp2_2
		global wp2_3
		global bias2_1
		global bias2_2
		global bias2_3
		delta1=x3-onehotenc
		tempx=np.zeros(shape=(x2.shape[0],x2.shape[1]))
		tempx=np.where(x2>0,1,x2)
		w_d1_pro=np.matmul(wp2_3,delta1.transpose())
		delta2=np.multiply(tempx.transpose(),w_d1_pro)
		temp2x=np.zeros(shape=(x1.shape[0],x1.shape[1]))
		temp2x=np.where(x1>0,1,x1)
		w_d2_pro=np.matmul(wp2_2,delta2)
		delta3=np.multiply(temp2x.transpose(),w_d2_pro)

		w3temp=np.zeros(shape=(32,3))
		x2ver=np.zeros(shape=(32,1))
		d1hor=np.zeros(shape=(1,3))
		b3temp=np.zeros(3)

		for i in range(minibatch.shape[0]):
			x2ver[:,0]=x2[i,:].transpose()
			d1hor[0,:]=delta1[i,:]
			w3temp=w3temp+wp2_3-0.01*np.matmul(x2ver,d1hor)
			b3temp=b3temp+bias2_3-0.01*delta1[i,:]

		wp2_3=w3temp/minibatch.shape[0]
		bias2_3=b3temp/minibatch.shape[0]

		w2temp=np.zeros(shape=(64,32))
		b2temp=np.zeros(32)
		x1ver=np.zeros(shape=(64,1))
		d2hor=np.zeros(shape=(1,32))
		delta2=delta2.transpose()
		for i in range(minibatch.shape[0]):
			x1ver[:,0]=x1[i,:].transpose()
			d2hor[0,:]=delta2[i,:]
			w2temp=w2temp+wp2_2-0.01*np.matmul(x1ver,d2hor)
			b2temp=b2temp+bias2_2-0.01*delta2[i,:]

		wp2_2=w2temp/minibatch.shape[0]
		bias2_2=b2temp/minibatch.shape[0]

		w1temp=np.zeros(shape=(7,64))
		b1temp=np.zeros(64)
		minitemp=minibatch.transpose()
		miniver=np.zeros(shape=(7,1))
		d3hor=np.zeros(shape=(1,64))
		delta3=delta3.transpose()
		for i in range(minibatch.shape[0]):
			miniver[:,0]=minitemp[:,i]
			d3hor[0,:]=delta3[i,:]
			w1temp=w1temp+wp2_1-0.01*np.matmul(miniver,d3hor)
			b1temp=b1temp+bias2_1-0.01*delta3[i,:]

		wp2_1=w1temp/minibatch.shape[0]
		bias2_1=b1temp/minibatch.shape[0]


#Training function
def Training(part):
	global train
	global onehotvec
	if part==1:
		for i in range(200):
			if i%10==0:
				Predict(part)
			k=0
			for j in range(5):
				minibatch=np.zeros(shape=(32,7))
				minibatch[:,:]=train[k:k+32,0:7]
				x1ans,x2ans=forward(minibatch,part)
				backward(minibatch,onehotvec[k:k+32,:],x1ans,x2ans,0,part)
				k=k+32
			minibatchfinal=np.zeros(shape=(8,7))
			minibatchfinal[:,:]=train[160:168,0:7]
			x1ans,x2ans=forward(minibatchfinal,part)
			backward(minibatchfinal,onehotvec[160:168],x1ans,x2ans,0,part)
	if part==2:
		for i in range(200):
			if i%10==0:
				Predict(part)
			k=0
			
			for j in range(5):
				
				minibatch=np.zeros(shape=(32,7))
				minibatch[:,:]=train[k:k+32,0:7]
				x1ans,x2ans,x3ans=forward(minibatch,part)
				backward(minibatch,onehotvec[k:k+32,:],x1ans,x2ans,x3ans,part)
				k=k+32
			minibatchfinal=np.zeros(shape=(8,7))
			minibatchfinal[:,:]=train[160:168,0:7]
			x1ans,x2ans,x3ans=forward(minibatchfinal,part)
			backward(minibatchfinal,onehotvec[160:168],x1ans,x2ans,x3ans,part)

#Predict function
def Predict(part):
	global part1Atrainacc
	global part1Atestacc
	global part1Btrainacc
	global part1Btestacc
	if part==1:
		x1,x2=forward(train[:,0:7],part)
		output=np.zeros(train.shape[0])
		for i in range(train.shape[0]):
			if x2[i,0]>x2[i,1] and x2[i,0]>x2[i,2]:
				output[i]=1
			elif x2[i,1]>x2[i,0] and x2[i,1]>x2[i,2]:
				output[i]=2
			elif x2[i,2]>x2[i,0] and x2[i,2]>x2[i,1]:
				output[i]=3

		count=0
		for i in range(train.shape[0]):
			if output[i]==train[i,7]:
				count=count+1

		part1Atrainacc.append(count/train.shape[0])
		x1,x2=forward(test[:,0:7],part)
		output=np.zeros(test.shape[0])
		for i in range(test.shape[0]):
			if x2[i,0]>x2[i,1] and x2[i,0]>x2[i,2]:
				output[i]=1
			elif x2[i,1]>x2[i,0] and x2[i,1]>x2[i,2]:
				output[i]=2
			elif x2[i,2]>x2[i,0] and x2[i,2]>x2[i,1]:
				output[i]=3

		count=0
		for i in range(test.shape[0]):
			if output[i]==test[i,7]:
				count=count+1

		part1Atestacc.append(count/test.shape[0])
		
		
	if part==2:
		x1,x2,x3=forward(train[:,0:7],part)
		output=np.zeros(train.shape[0])
		for i in range(train.shape[0]):
			if x3[i,0]>x3[i,1] and x3[i,0]>x3[i,2]:
				output[i]=1
			elif x3[i,1]>x3[i,0] and x3[i,1]>x3[i,2]:
				output[i]=2
			elif x3[i,2]>x3[i,0] and x3[i,2]>x3[i,1]:
				output[i]=3

		count=0
		for i in range(train.shape[0]):
			if output[i]==train[i,7]:
				count=count+1

		part1Btrainacc.append(count/train.shape[0])
	
		x1,x2,x3=forward(test[:,0:7],part)
		output=np.zeros(test.shape[0])
		for i in range(test.shape[0]):
			if x3[i,0]>x3[i,1] and x3[i,0]>x3[i,2]:
				output[i]=1
			elif x3[i,1]>x3[i,0] and x3[i,1]>x3[i,2]:
				output[i]=2
			elif x3[i,2]>x3[i,0] and x3[i,2]>x3[i,1]:
				output[i]=3

		count=0
		for i in range(test.shape[0]):
			if output[i]==test[i,7]:
				count=count+1

		part1Btestacc.append(count/test.shape[0])
		
		


#Some variables declared
wp1_1=np.zeros(shape=(7,32))
wp1_2=np.zeros(shape=(32,3))
bias1_1=np.zeros(32)
bias1_2=np.zeros(3)
wp2_1=np.zeros(shape=(7,64))
wp2_2=np.zeros(shape=(64,32))
wp2_3=np.zeros(shape=(32,3))
bias2_1=np.zeros(64)
bias2_2=np.zeros(32)
bias2_3=np.zeros(3)
train=np.zeros(shape=(168,8))
test=np.zeros(shape=(42,8))

#Accuracy lists
part1Atrainacc=[]
part1Atestacc=[]
part1Btrainacc=[]
part1Btestacc=[]
iterations=[0,10,20,30,40,50,60,70,80,90,100
,110,120,130,140,150,160,170,180,190,200]

#Preprocess the data
Preprocess()
#load the train and test dataset
DataLoader()
#compute the one hot encoding vectors
onehotvec=np.zeros(shape=(168,3))
v1=[1,0,0]
v2=[0,1,0]
v3=[0,0,1]
for i in range(168):
	if train[i,7]==1:
		onehotvec[i,:]=v1
	if train[i,7]==2:
		onehotvec[i,:]=v2
	if train[i,7]==3:
		onehotvec[i,:]=v3

#for specification 1A
weightinitializer(1)
Training(1)

Predict(1)
print('Part 1A:')
print('Training Accuracy :',part1Atrainacc[20])
print('Testing Accuracy :',part1Atestacc[20])
print(' ')

#Plotting Data
plt.plot(iterations,part1Atrainacc,'b-',label='Training Accuracy',marker='.')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs No of Epochs (PART 1A)')
plt.plot(iterations,part1Atestacc,color='green',label='Testing Accuracy',marker='.')
plt.legend(loc='best')
#Saving plot
plt.savefig('Part1A.png')
plt.clf()

#for specification 1B
weightinitializer(2)
Training(2)
Predict(2)

print('Part 1B:')
print('Training Accuracy :',part1Btrainacc[20])
print('Testing Accuracy :',part1Btestacc[20])
print(' ')

#Plotting Data
plt.plot(iterations,part1Btrainacc,'b-',label='Training Accuracy',marker='.')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs No of Epochs (PART 1B)')
plt.plot(iterations,part1Btestacc,color='green',label='Testing Accuracy',marker='.')
plt.legend(loc='best')
#Saving plot
plt.savefig('Part1B.png')
plt.clf()


