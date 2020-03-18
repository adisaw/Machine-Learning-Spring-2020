import numpy as np
import matplotlib.pyplot as plt


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

traindata = np.genfromtxt('train.csv', dtype=np.float64, delimiter=',', skip_header=1) 
trainX=np.asarray(traindata[:,0])
trainY=np.asarray(traindata[:,1])
testdata = np.genfromtxt('test.csv', dtype=np.float64, delimiter=',', skip_header=1) 
testX=np.asarray(testdata[:,0])
testY=np.asarray(testdata[:,1])

plt.scatter(trainX,trainY,label='Training data')
plt.xlabel('Feature')
plt.ylabel('Label')
plt.title('Plot of Degree1 equation')

coef1=np.loadtxt('para1deg.txt',dtype=np.float64)
coef2=np.loadtxt('para2deg.txt',dtype=np.float64)
coef3=np.loadtxt('para3deg.txt',dtype=np.float64)
coef4=np.loadtxt('para4deg.txt',dtype=np.float64)
coef5=np.loadtxt('para5deg.txt',dtype=np.float64)
coef6=np.loadtxt('para6deg.txt',dtype=np.float64)
coef7=np.loadtxt('para7deg.txt',dtype=np.float64)
coef8=np.loadtxt('para8deg.txt',dtype=np.float64)
coef9=np.loadtxt('para9deg.txt',dtype=np.float64)

xval=np.linspace(0, 1, 500)
#print(xval)

vals=np.zeros(shape=(len(xval),10))
for i in range (len(xval)):
	vals[i][0]=1
	tempval=1
	for j in range (1,10):
		tempval=tempval*xval[i]
		vals[i][j]=tempval

deg1yval=np.matmul(vals[:,0:2],np.transpose(coef1))
plt.scatter(xval,deg1yval,color='green',label='Degree 1 Equation')
plt.legend(loc='best')
plt.savefig('degree1.png')
plt.clf()

plt.scatter(trainX,trainY,label='Training data')
plt.xlabel('Feature')
plt.ylabel('Label')
plt.title('Plot of Degree2 equation')
deg2yval=np.matmul(vals[:,0:3],np.transpose(coef2))
plt.scatter(xval,deg2yval,color='green',label='Degree 2 Equation')
plt.legend(loc='best')
plt.savefig('degree2.png')
plt.clf()

plt.scatter(trainX,trainY,label='Training data')
plt.xlabel('Feature')
plt.ylabel('Label')
plt.title('Plot of Degree3 equation')
deg3yval=np.matmul(vals[:,0:4],np.transpose(coef3))
plt.scatter(xval,deg3yval,color='green',label='Degree 3 Equation')
plt.legend(loc='best')
plt.savefig('degree3.png')
plt.clf()

plt.scatter(trainX,trainY,label='Training data')
plt.xlabel('Feature')
plt.ylabel('Label')
plt.title('Plot of Degree4 equation')
deg4yval=np.matmul(vals[:,0:5],np.transpose(coef4))
plt.scatter(xval,deg4yval,color='green',label='Degree 4 equation')
plt.legend(loc='best')
plt.savefig('degree4.png')
plt.clf()

plt.scatter(trainX,trainY,label='Training data')
plt.xlabel('Feature')
plt.ylabel('Label')
plt.title('Plot of Degree5 equation')
deg5yval=np.matmul(vals[:,0:6],np.transpose(coef5))
plt.scatter(xval,deg5yval,color='green',label='Degree 5 Equation')
plt.legend(loc='best')
plt.savefig('degree5.png')
plt.clf()

plt.scatter(trainX,trainY,label='Training data')
plt.xlabel('Feature')
plt.ylabel('Label')
plt.title('Plot of Degree6 equation')
deg6yval=np.matmul(vals[:,0:7],np.transpose(coef6))
plt.scatter(xval,deg6yval,color='green',label='Degree 6 Equation')
plt.legend(loc='best')
plt.savefig('degree6.png')
plt.clf()

plt.scatter(trainX,trainY,label='Training data')
plt.xlabel('Feature')
plt.ylabel('Label')
plt.title('Plot of Degree7 equation')
deg7yval=np.matmul(vals[:,0:8],np.transpose(coef7))
plt.scatter(xval,deg7yval,color='green',label='Degree 7 Equation')
plt.legend(loc='best')
plt.savefig('degree7.png')
plt.clf()

plt.scatter(trainX,trainY,label='Training data')
plt.xlabel('Feature')
plt.ylabel('Label')
plt.title('Plot of Degree8 equation')
deg8yval=np.matmul(vals[:,0:9],np.transpose(coef8))
plt.scatter(xval,deg8yval,color='green',label='Degree 8 Equation')
plt.legend(loc='best')
plt.savefig('degree8.png')
plt.clf()

plt.scatter(trainX,trainY,label='Training data')
plt.xlabel('Feature')
plt.ylabel('Label')
plt.title('Plot of Degree9 equation')
deg9yval=np.matmul(vals[:,0:10],np.transpose(coef9))
plt.scatter(xval,deg9yval,color='green',label='Degree 9 Equation')
plt.legend(loc='best')
plt.savefig('degree9.png')
plt.clf()

vals2=np.zeros(shape=(len(trainX),10))
for i in range (len(trainX)):
	vals2[i][0]=1
	tempval=1
	for j in range (1,10):
		tempval=tempval*trainX[i]
		vals2[i][j]=tempval

vals3=np.zeros(shape=(len(testX),10))
for i in range (len(testX)):
	vals3[i][0]=1
	tempval=1
	for j in range (1,10):
		tempval=tempval*testX[i]
		vals3[i][j]=tempval

testerror=np.zeros(9)
trainerror=np.zeros(9)

trainerror[0]=errr(1,vals2,trainY,coef1)
trainerror[1]=errr(2,vals2,trainY,coef2)
trainerror[2]=errr(3,vals2,trainY,coef3)
trainerror[3]=errr(4,vals2,trainY,coef4)
trainerror[4]=errr(5,vals2,trainY,coef5)
trainerror[5]=errr(6,vals2,trainY,coef6)
trainerror[6]=errr(7,vals2,trainY,coef7)
trainerror[7]=errr(8,vals2,trainY,coef8)
trainerror[8]=errr(9,vals2,trainY,coef9)
np.savetxt("trainingerror.txt",trainerror)

testerror[0]=errr(1,vals3,testY,coef1)
testerror[1]=errr(2,vals3,testY,coef2)
testerror[2]=errr(3,vals3,testY,coef3)
testerror[3]=errr(4,vals3,testY,coef4)
testerror[4]=errr(5,vals3,testY,coef5)
testerror[5]=errr(6,vals3,testY,coef6)
testerror[6]=errr(7,vals3,testY,coef7)
testerror[7]=errr(8,vals3,testY,coef8)
testerror[8]=errr(9,vals3,testY,coef9)
np.savetxt("testerror.txt",testerror)


x=np.linspace(1,9,9)
plt.plot(x,trainerror,'b-',label='Training Error',marker='.')
plt.xlabel('Degree')
plt.ylabel('Squared Error')
plt.title('Squared Error vs Degree')

plt.plot(x,testerror,color='green',label='Test Error',marker='.')
plt.legend(loc='best')
plt.savefig('ErrorvsDegree.png')

plt.clf()
