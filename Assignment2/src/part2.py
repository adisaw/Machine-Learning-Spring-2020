import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
#get data from csv
data = np.genfromtxt('logisticregression.csv', dtype=np.float32, delimiter=',')
#sigmoid function
def sigmoid(val):
 	return 1/(1+np.exp(-val))

#error function
def errr(inputs, Y_val, w):
	erval=0
	m=len(Y_val)
	X_val=np.zeros(len(Y_val))
	one=np.ones(len(Y_val))
	first=np.zeros(len(Y_val))
	second=np.zeros(len(Y_val))
	wtr=np.transpose(w)
	X_temp=np.matmul(inputs[:,0:12],wtr)
	X_temp=np.transpose(X_temp)
	h=sigmoid(X_temp)
	first=np.dot(Y_val,np.log(h))
	htemp=np.subtract(one,h)
	ytemp=np.subtract(one,Y_val)
	second=np.dot(np.log(htemp),ytemp)
	final=np.add(first,second)
	erval=np.sum(final)
	return -erval/m

#helper function
def helper(k,inputs,Y_val,w):
	m=len(Y_val)
	ans=0
	X_val=np.zeros(len(Y_val))
	wtr=np.transpose(w)
	X_temp=np.matmul(inputs[:,0:12],wtr)
	X_new=np.transpose(X_temp)
	h=sigmoid(X_new)
	X_final=np.subtract(h,Y_val)
	ans=np.matmul(X_final,inputs[:,k])
	return ans
#logistic regression function
def logistic_regression(alpha,inputs,Y_val):
	w=np.random.randn(12)
	temp_w=np.zeros(12)
	m=len(Y_val)
	itrcount=0
	preverr=errr(inputs,Y_val,w)
	while 1:
 		if itrcount%1000==0 :
 			print(itrcount)
 		for i in range (12):
 			value = helper(i,inputs,Y_val,w)
 			temp_w[i]=w[i]-(alpha*value)/m

 		w=np.copy(temp_w)
 		errorf=errr(inputs,Y_val,w)

 		#convergence condition
 		if abs(errorf-preverr)<0.0000001:
 			break
 		else:
 			preverr=errorf
 			itrcount=itrcount+1
	#returning parameters
	return w
 	



#shuffle data
np.random.shuffle(data)
Y_val=np.zeros(len(data))
Y_val=data[:,11]

#divide data into three sets

data1=np.zeros(shape=(533,12))
data1y=np.zeros(533)

for i in range(data1.shape[0]) :
	data1[i][0]=1
data1[:,1:12]=data[0:533,0:11]
data1y=data[0:533,11]

data2=np.zeros(shape=(533,12))
data2y=np.zeros(533)
for i in range(data2.shape[0]):
	data2[i][0]=1
data2[:,1:12]=data[533:1066,0:11]
data2y=data[533:1066,11]

data3=np.zeros(shape=(533,12))
data3y=np.zeros(533)
for i in range(data3.shape[0]):
	data3[i][0]=1
data3[:,1:12]=data[1066:1599,0:11]
data3y=data[1066:1599,11]



accuracy=0
precision=0
recall=0
accuracyscikit=0
precisionscikit=0
recallscikit=0

#train: data1,data2 test:data3
trainX=np.zeros(shape=(1066,12))
trainY=np.zeros(1066)
testX=np.zeros(shape=(533,12))
testY=np.zeros(533)
trainX[0:533,:]=data1
trainX[533:1066,:]=data2
trainY[0:533]=data1y
trainY[533:1066]=data2y
testX=data3
testY=data3y
params=np.zeros(1066)
#scikit-learn logistic regression
clf=LogisticRegression(solver='saga',penalty='none')
clf.fit(trainX,trainY)
predictions=clf.predict(testX)
#getting accuracy, recall and precision
accuracyscikit=accuracyscikit+accuracy_score(testY,predictions)
precisionscikit=precisionscikit+precision_score(testY,predictions)
recallscikit=recallscikit+recall_score(testY,predictions)
#user defined logisticregression
params=logistic_regression(0.05,trainX,trainY)
print(" ")
output_Y=np.zeros(533)
params=np.transpose(params)
output_Y=sigmoid(np.matmul(testX,params))
predict_Y=np.zeros(533)
for i in range(533):
	if output_Y[i]>0.5:
		predict_Y[i]=1
	else:
		predict_Y[i]=0
count=0
truepositive=0
falsepositive=0
falsenegative=0
#calculates truepositive, falsepositive, falsenegative
for i in range(533):
	if predict_Y[i]==testY[i]:
		count=count+1
	if predict_Y[i]==1 and testY[i]==1:
		truepositive=truepositive+1
	if predict_Y[i]==1 and testY[i]==0:
		falsepositive=falsepositive+1
	if predict_Y[i]==0 and testY[i]==1:
		falsenegative=falsenegative+1



accuracy=accuracy+count/533
precision=precision+(truepositive)/(truepositive+falsepositive)
recall=recall+(truepositive)/(truepositive+falsenegative)

#train: data2,data3 test:data1
trainX[0:533,:]=data2
trainX[533:1066,:]=data3
trainY[0:533]=data2y
trainY[533:1066]=data3y
testX=data1
testY=data1y
params=np.zeros(1066)
#scikit-learn logistic regression
clf.fit(trainX,trainY)
predictions=clf.predict(testX)
#getting accuracy, recall and precision
accuracyscikit=accuracyscikit+accuracy_score(testY,predictions)
precisionscikit=precisionscikit+precision_score(testY,predictions)
recallscikit=recallscikit+recall_score(testY,predictions)
#user defined logisticregression
params=logistic_regression(0.05,trainX,trainY)
print(" ");
output_Y=np.zeros(533)
params=np.transpose(params)
output_Y=sigmoid(np.matmul(testX,params))
predict_Y=np.zeros(533)
for i in range(533):
	if output_Y[i]>0.5:
		predict_Y[i]=1
	else:
		predict_Y[i]=0
count=0
truepositive=0
falsepositive=0
falsenegative=0
#calculates truepositive, falsepositive, falsenegative
for i in range(533):
	if predict_Y[i]==testY[i]:
		count=count+1
	if predict_Y[i]==1 and testY[i]==1:
		truepositive=truepositive+1
	if predict_Y[i]==1 and testY[i]==0:
		falsepositive=falsepositive+1
	if predict_Y[i]==0 and testY[i]==1:
		falsenegative=falsenegative+1


accuracy=accuracy+count/533
precision=precision+(truepositive)/(truepositive+falsepositive)
recall=recall+(truepositive)/(truepositive+falsenegative)

trainX[0:533,:]=data3
trainX[533:1066,:]=data1
trainY[0:533]=data3y
trainY[533:1066]=data1y
testX=data2
testY=data2y
params=np.zeros(1066)
#scikit-learn logistic regression
clf.fit(trainX,trainY)
predictions=clf.predict(testX)
#getting accuracy, recall and precision
accuracyscikit=accuracyscikit+accuracy_score(testY,predictions)
precisionscikit=precisionscikit+precision_score(testY,predictions)
recallscikit=recallscikit+recall_score(testY,predictions)
#user defined logisticregression
params=logistic_regression(0.05,trainX,trainY)
output_Y=np.zeros(533)
params=np.transpose(params)
output_Y=sigmoid(np.matmul(testX,params))
predict_Y=np.zeros(533)
for i in range(533):
	if output_Y[i]>0.5:
		predict_Y[i]=1
	else:
		predict_Y[i]=0
count=0
truepositive=0
falsepositive=0
falsenegative=0
#calculates truepositive, falsepositive, falsenegative
for i in range(533):
	if predict_Y[i]==testY[i]:
		count=count+1
	if predict_Y[i]==1 and testY[i]==1:
		truepositive=truepositive+1
	if predict_Y[i]==1 and testY[i]==0:
		falsepositive=falsepositive+1
	if predict_Y[i]==0 and testY[i]==1:
		falsenegative=falsenegative+1

accuracy=accuracy+count/533
precision=precision+(truepositive)/(truepositive+falsepositive)
recall=recall+(truepositive)/(truepositive+falsenegative)
#getting mean values
accuracy=accuracy/3
precision=precision/3
recall=recall/3

accuracyscikit=accuracyscikit/3
precisionscikit=precisionscikit/3
recallscikit=recallscikit/3
#Accuracy , Precision , Recall of model without using scikit-learn
print('Mean Accuracy  :', accuracy)
print('Mean Precision :', precision)
print('Mean Recall :',recall)
#Accuracy , Precision , Recall of model using scikit-learn
print('Mean Accuracy using scikit-learn :', accuracyscikit)
print('Mean Precision scikit-learn :', precisionscikit)
print('Mean Recall scikit-learn :',recallscikit)










