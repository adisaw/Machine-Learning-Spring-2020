import numpy as np 
import warnings
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings('ignore')
train=np.zeros(shape=(168,8))
test=np.zeros(shape=(42,8))

#Load train and test dataset
train=np.genfromtxt('train.csv',dtype=np.float64,delimiter=',')
test=np.genfromtxt('test.csv',dtype=np.float64,delimiter=',')

#initializing some variables
trainX=np.zeros(shape=(168,7))
trainX[:,:]=train[:,0:7]
trainY=np.zeros(168)
trainY=train[:,7]
testX=np.zeros(shape=(42,7))
testX[:,:]=test[:,0:7]
testY=np.zeros(42)
testY=test[:,7]

#Specification 1A 32 neurons in hidden layer
mlp1=MLPClassifier(hidden_layer_sizes=(32),activation='logistic',
	solver='sgd',batch_size=32,learning_rate_init=0.01,max_iter=200)
mlp1.fit(trainX,trainY)
output=mlp1.predict(trainX)
#Printing Accuracy
count=0
for i in range(trainY.shape[0]):
	if trainY[i]==output[i]:
		count=count+1
print('Part 2 Specification 1A:')
print('Training Accuracy :',count/trainY.shape[0])

output=mlp1.predict(testX)
count=0
for i in range(testY.shape[0]):
	if testY[i]==output[i]:
		count=count+1

print('Testing Accuracy :',count/testY.shape[0])


#Specification 1A 64,32 neurons in hidden layer
mlp2=MLPClassifier(hidden_layer_sizes=(64,32),activation='relu',solver='sgd',
	batch_size=32,learning_rate_init=0.01,max_iter=200)

mlp2.fit(trainX,trainY)
output=mlp2.predict(trainX)
#Printing Accuracy
count=0
for i in range(trainY.shape[0]):
	if trainY[i]==output[i]:
		count=count+1
print(' ')
print('Part 2 Specification 1B')
print('Training Accuracy :',count/trainY.shape[0])

output=mlp2.predict(testX)
count=0
for i in range(testY.shape[0]):
	if testY[i]==output[i]:
		count=count+1

print('Testing Accuracy :',count/testY.shape[0])
