import numpy as np
import matplotlib.pyplot as plt

traindata = np.genfromtxt('train.csv', dtype=np.float64, delimiter=',', skip_header=1) 
trainX=np.asarray(traindata[:,0])
trainY=np.asarray(traindata[:,1])
testdata = np.genfromtxt('test.csv', dtype=np.float64, delimiter=',', skip_header=1) 
testX=np.asarray(testdata[:,0])
testY=np.asarray(testdata[:,1])

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

def errlasso(n,vals,Y_val,w,lambdas):
	erval=0
	m=len(Y_val)
	X_val=np.zeros(len(Y_val))
	wtr=np.transpose(w)
	X_temp=np.matmul(vals[:,0:n+1],wtr)

	X_val=np.transpose(X_temp)
	X_val=np.subtract(X_val,Y_val)
	X_val=np.square(X_val)
	erval=np.sum(X_val);
	ferr=(erval+lambdas*np.sum(w))*0.5/m
	return ferr


def errridge(n,vals,Y_val,w,lambdas):
	erval=0
	m=len(Y_val)
	X_val=np.zeros(len(Y_val))
	wtr=np.transpose(w)
	X_temp=np.matmul(vals[:,0:n+1],wtr)

	X_val=np.transpose(X_temp)
	X_val=np.subtract(X_val,Y_val)
	X_val=np.square(X_val)
	erval=np.sum(X_val);
	w=np.power(w,2)
	ferr=(erval+ lambdas*np.sum(w))*0.5/m
	return ferr

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

def linear_lassoreg(n,alpha,vals,Y_val,lambdas):
	#w=np.random.randn(n+1)
	w=np.zeros(n+1)
	temp_w=np.zeros(n+1)
	m=len(Y_val)
	itrcount=0
	preverr=errlasso(n,vals,Y_val,w,lambdas)
	while itrcount<100000:
		if itrcount%1000==0 :
			print(itrcount)
		for i in range (n+1):
			value=helper(n,i,vals,Y_val,w)
			temp_w[i]=w[i]-((alpha*value)/m) -alpha*lambdas*0.5/m
	
		w=np.copy(temp_w)
		errorf=errlasso(n,vals,Y_val,w,lambdas)

		if abs(errorf-preverr)<0.00000001:
			break
		else:
			preverr=errorf
			itrcount=itrcount+1
	return w

def linear_ridgereg(n,alpha,vals,Y_val,lambdas):
	#w=np.random.randn(n+1)
	w=np.zeros(n+1)
	temp_w=np.zeros(n+1)
	m=len(Y_val)
	itrcount=0
	preverr=errridge(n,vals,Y_val,w,lambdas)
	while itrcount<100000:
		if itrcount%1000==0 :
			print(itrcount)
		for i in range (n+1):
			value=helper(n,i,vals,Y_val,w)
			temp_w[i]=w[i]-((alpha*value)/m) -alpha*lambdas*w[i]/m
	
		w=np.copy(temp_w)
		errorf=errridge(n,vals,Y_val,w,lambdas)

		if abs(errorf-preverr)<0.00000001:
			break
		else:
			preverr=errorf
			itrcount=itrcount+1
	return w

vals=np.zeros(shape=(len(trainX),10))
for i in range (len(trainX)):
	vals[i][0]=1
	tempval=1
	for j in range (1,10):
		tempval=tempval*trainX[i]
		vals[i][j]=tempval

vals2=np.zeros(shape=(len(testX),10))
for i in range (len(testX)):
	vals2[i][0]=1
	tempval=1
	for j in range (1,10):
		tempval=tempval*testX[i]
		vals2[i][j]=tempval


coefl1_25=np.zeros(2)
coefl1_5=np.zeros(2)
coefl1_75=np.zeros(2)
coefl1_1=np.zeros(2)

coefr1_25=np.zeros(2)
coefr1_5=np.zeros(2)
coefr1_75=np.zeros(2)
coefr1_1=np.zeros(2)

coefl1_25=linear_lassoreg(1,0.05,vals,trainY,0.25)
coefl1_5=linear_lassoreg(1,0.05,vals,trainY,0.50)
coefl1_75=linear_lassoreg(1,0.05,vals,trainY,0.75)
coefl1_1=linear_lassoreg(1,0.05,vals,trainY,1)

coefr1_25=linear_ridgereg(1,0.05,vals,trainY,0.25)
coefr1_5=linear_ridgereg(1,0.05,vals,trainY,0.50)
coefr1_75=linear_ridgereg(1,0.05,vals,trainY,0.75)
coefr1_1=linear_ridgereg(1,0.05,vals,trainY,1)

coefl9_25=np.zeros(10)
coefl9_5=np.zeros(10)
coefl9_75=np.zeros(10)
coefl9_1=np.zeros(10)

coefr9_25=np.zeros(10)
coefr9_5=np.zeros(10)
coefr9_75=np.zeros(10)
coefr9_1=np.zeros(10)

coefl9_25=linear_lassoreg(9,0.05,vals,trainY,0.25)
coefl9_5=linear_lassoreg(9,0.05,vals,trainY,0.50)
coefl9_75=linear_lassoreg(9,0.05,vals,trainY,0.75)
coefl9_1=linear_lassoreg(9,0.05,vals,trainY,1)

coefr9_25=linear_ridgereg(9,0.05,vals,trainY,0.25)
coefr9_5=linear_ridgereg(9,0.05,vals,trainY,0.50)
coefr9_75=linear_ridgereg(9,0.05,vals,trainY,0.75)
coefr9_1=linear_ridgereg(9,0.05,vals,trainY,1)

#print(coefr9_1)
#print(coefl1_25)
#print(coefl1_75)
#print(coefr1_25)
#print(coefr1_75)

##print(coefl9_25)
#print(coefl9_75)
#print(coefr9_25)
#print(coefr9_75)

errl1train=np.zeros(4)
errl1train[0]=errr(1,vals,trainY,coefl1_25)
errl1train[1]=errr(1,vals,trainY,coefl1_5)
errl1train[2]=errr(1,vals,trainY,coefl1_75)
errl1train[3]=errr(1,vals,trainY,coefl1_1)

errr1train=np.zeros(4)
errr1train[0]=errr(1,vals,trainY,coefr1_25)
errr1train[1]=errr(1,vals,trainY,coefr1_5)
errr1train[2]=errr(1,vals,trainY,coefr1_75)
errr1train[3]=errr(1,vals,trainY,coefr1_1)

errl1test=np.zeros(4)
errl1test[0]=errr(1,vals2,testY,coefl1_25)
errl1test[1]=errr(1,vals2,testY,coefl1_5)
errl1test[2]=errr(1,vals2,testY,coefl1_75)
errl1test[3]=errr(1,vals2,testY,coefl1_1)

errr1test=np.zeros(4)
errr1test[0]=errr(1,vals2,testY,coefr1_25)
errr1test[1]=errr(1,vals2,testY,coefr1_5)
errr1test[2]=errr(1,vals2,testY,coefr1_75)
errr1test[3]=errr(1,vals2,testY,coefr1_1)

x=np.linspace(0.25,1,4)

plt.plot(x,errl1train,color='blue',label='Training Error Lasso Regression',marker='.')
plt.xlabel('Lambda value')
plt.ylabel('Error')
plt.title('Error vs Lambda value Degree 1 Lasso Regression')
plt.plot(x,errl1test,color='green',label='Test Error Lasso Regression',marker='.')

plt.legend(loc='best')
plt.savefig('Degree 1 Lasso Regression.png')
plt.clf()


plt.plot(x,errr1train,color='blue',label='Training Error Ridge Regression',marker='.')
plt.plot(x,errr1test,color='green',label='Test Error Ridge Regression',marker='.')
plt.xlabel('Lambda value')
plt.ylabel('Error')
plt.title('Error vs Lambda value Degree 1 Ridge Regression')
plt.legend(loc='best')
plt.savefig('Degree 1 Ridge Regression.png')
plt.clf()

errl9train=np.zeros(4)
errl9train[0]=errr(9,vals,trainY,coefl9_25)
errl9train[1]=errr(9,vals,trainY,coefl9_5)
errl9train[2]=errr(9,vals,trainY,coefl9_75)
errl9train[3]=errr(9,vals,trainY,coefl9_1)

errr9train=np.zeros(4)
errr9train[0]=errr(9,vals,trainY,coefr9_25)
errr9train[1]=errr(9,vals,trainY,coefr9_5)
errr9train[2]=errr(9,vals,trainY,coefr9_75)
errr9train[3]=errr(9,vals,trainY,coefr9_1)

errl9test=np.zeros(4)
errl9test[0]=errr(9,vals2,testY,coefl9_25)
errl9test[1]=errr(9,vals2,testY,coefl9_5)
errl9test[2]=errr(9,vals2,testY,coefl9_75)
errl9test[3]=errr(9,vals2,testY,coefl9_1)

errr9test=np.zeros(4)
errr9test[0]=errr(9,vals2,testY,coefr9_25)
errr9test[1]=errr(9,vals2,testY,coefr9_5)
errr9test[2]=errr(9,vals2,testY,coefr9_75)
errr9test[3]=errr(9,vals2,testY,coefr9_1)



plt.plot(x,errl9train,'blue',label='Training Error Lasso Regression',marker='.')
plt.xlabel('Lambda value')
plt.ylabel('Error')
plt.title('Error vs Lambda value Degree 9 Lasso Regression')
plt.plot(x,errl9test,color='green',label='Test Error Lasso Regression',marker='.')

plt.legend(loc='best')
plt.savefig('Degree 9 Lasso Regression.png')
plt.clf()

plt.plot(x,errr9train,'blue',label='Training Error Ridge Regression',marker='.')
plt.xlabel('Lambda value')
plt.ylabel('Error')
plt.title('Error vs Lambda value Degree 9 Ridge Regression')
plt.plot(x,errr9test,color='green',label='Test Error Ridge Regression',marker='.')

plt.legend(loc='best')
plt.savefig('Degree 9 Ridge Regression.png')
plt.clf()




















