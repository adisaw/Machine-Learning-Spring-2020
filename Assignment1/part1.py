import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

traindata = np.genfromtxt('train.csv', dtype=np.float64, delimiter=',', skip_header=1) 
trainX=np.asarray(traindata[:,0])
trainY=np.asarray(traindata[:,1])
testdata = np.genfromtxt('test.csv', dtype=np.float64, delimiter=',', skip_header=1) 
testX=np.asarray(testdata[:,0])
testY=np.asarray(testdata[:,1])
#print(testY)
#print(testX)
#print(df_csv)
def errr(n,vals,Y_val,w):
	erval=0
	m=len(Y_val)
	X_val=np.zeros(len(Y_val))
	wtr=np.transpose(w)
	X_temp=np.matmul(vals[:,0:n+1],wtr)

	X_val=np.transpose(X_temp)
	X_val=np.subtract(X_val,Y_val)
	X_val=np.square(X_val)
	erval=np.sum(X_val);
	
		
	
	return (erval*0.5)/m

def helper(n,k,vals,Y_val,w):
	m=len(Y_val)
	ans=0
	X_val=np.zeros(len(Y_val))
	wtr=np.transpose(w)
	X_temp=np.matmul(vals[:,0:n+1],wtr)
	X_new=np.transpose(X_temp)
	X_new=np.subtract(X_new,Y_val)

	ans=np.matmul(X_new,vals[:,k])
		
	return ans

def linear_reg(n,alpha,vals,Y_val):
	w=np.random.randn(n+1)
	temp_w=np.zeros(n+1)
	m=len(Y_val)
	itrcount=0
	preverr=errr(n,vals,Y_val,w)
	while 1:
		if itrcount%1000==0 :
			print(itrcount)
		for i in range (n+1):
			value=helper(n,i,vals,Y_val,w)
			temp_w[i]=w[i]-(alpha*value)/m
	
		w=np.copy(temp_w)
		errorf=errr(n,vals,Y_val,w)

		if abs(errorf-preverr)<0.00000001:
			break
		else:
			preverr=errorf
			itrcount=itrcount+1
	return w





plt.scatter(trainX,trainY)
plt.xlabel('Feature')
plt.ylabel('Label')
plt.title('Plot of Training set')
plt.savefig('trainset.png')
plt.clf() 
plt.scatter(testX,testY)
plt.xlabel('Feature')
plt.ylabel('Label')
plt.title('Plot of Test set')
plt.savefig('testset.png')








vals=np.zeros(shape=(len(trainX),10))
for i in range (len(trainX)):
	vals[i][0]=1
	tempval=1
	for j in range (1,10):
		tempval=tempval*trainX[i]
		vals[i][j]=tempval


#print(vals[:,0:2].shape)
coef1=np.zeros(2)
coef1=linear_reg(1,0.05,vals,trainY)
coef2=np.zeros(3)
coef2=linear_reg(2,0.05,vals,trainY)


#print(np.shape(vals[:,0:4]))
#print(np.shape(np.dot(coef2, (vals[:,0:4]).T)))
#print(np.shape(vals[:, 1]))


coef3=np.zeros(4)
coef3=linear_reg(3,0.05,vals,trainY)
coef4=np.zeros(5)
coef4=linear_reg(4,0.05,vals,trainY)
coef5=np.zeros(6)
coef5=linear_reg(5,0.05,vals,trainY)
coef6=np.zeros(7)
coef6=linear_reg(6,0.05,vals,trainY)
coef7=np.zeros(8)
coef7=linear_reg(7,0.05,vals,trainY)
coef8=np.zeros(9)
coef8=linear_reg(8,0.05,vals,trainY)
coef9=np.zeros(10)
coef9=linear_reg(9,0.05,vals,trainY)
print('Parameters for degree1 ')
print(coef1)
np.savetxt("para1deg.txt",coef1)
print('Parameters for degree2 ')
print(coef2)
np.savetxt("para2deg.txt",coef2)
print('Parameters for degree3 ')
print(coef3)
np.savetxt("para3deg.txt",coef3)
print('Parameters for degree4 ')
print(coef4)
np.savetxt("para4deg.txt",coef4)
print('Parameters for degree5 ')
print(coef5)
np.savetxt("para5deg.txt",coef5)
print('Parameters for degree6 ')
print(coef6)
np.savetxt("para6deg.txt",coef6)
print('Parameters for degree7 ')
print(coef7)
np.savetxt("para7deg.txt",coef7)
print('Parameters for degree8 ')
print(coef8)
np.savetxt("para8deg.txt",coef8)
print('Parameters for degree9 ')
print(coef9)
np.savetxt("para9deg.txt",coef9)

vals2=np.zeros(shape=(len(testX),10))
for i in range (len(testX)):
	vals2[i][0]=1
	tempval=1
	for j in range (1,10):
		tempval=tempval*testX[i]
		vals2[i][j]=tempval

err1= errr(1,vals2,testY,coef1)
print('Squared test error for degree1 equation :',err1)
err2= errr(2,vals2,testY,coef2)
print('Squared test error for degree2 equation :',err2)
err3= errr(3,vals2,testY,coef3)
print('Squared test error for degree3 equation :',err3)
err4= errr(4,vals2,testY,coef4)
print('Squared test error for degree4 equation :' ,err4)
err5= errr(5,vals2,testY,coef5)
print('Squared test error for degree5 equation :' ,err5)
err6= errr(6,vals2,testY,coef6)
print('Squared test error for degree6 equation :' ,err6)
err7= errr(7,vals2,testY,coef7)
print('Squared test error for degree7 equation :' ,err7)
err8= errr(8,vals2,testY,coef8)
print('Squared test error for degree8 equation :' ,err8)
err9= errr(9,vals2,testY,coef9)
print('Squared test error for degree9 equation :' ,err9)





#plt.scatter(vals[:, 1], np.dot(coef3, (vals[:,0:4]).T), color="red")
#plt.show()




