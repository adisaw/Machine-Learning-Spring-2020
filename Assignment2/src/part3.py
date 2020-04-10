import numpy as np
import math
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier
#get data from csv
data = np.genfromtxt('decisiontree.csv', dtype=np.float32, delimiter=',')

#creating decision tree object
class DT(object):
	def __init__(self):
		self.is_root=False
		self.attr=None
		self.num_children=0
		self.left=None
		self.leftmid=None
		self.rightmid=None
		self.right=None
		self.parent_attribute_val=None
		self.parent_attribute=None
		self.terminal_class=None
		self.is_terminal=False
		self.parent=None
		self.majclass=None
#splitnode function
def splitnode(attr,data):
	left=np.empty((0,12),dtype='float32')
	right=np.empty((0,12),dtype='float32')
	leftmid=np.empty((0,12),dtype='float32')
	rightmid=np.empty((0,12),dtype='float32')
	for row in data:
		if row[attr]==0:
			left=np.concatenate((left,np.array([row])))
		elif row[attr]==1:
			leftmid=np.concatenate((leftmid,np.array([row])))
		elif row[attr]==2:
			rightmid=np.concatenate((rightmid,np.array([row])))
		elif row[attr]==3:
			right=np.concatenate((right,np.array([row])))
	return left,leftmid,rightmid,right
#calculates info gain
def calcinfogain(data,groups):
	totalinst=0.0
	for group in groups:
		totalinst=totalinst+len(group)
	entropy=0.0
	for group in groups:
		score=0.0
		size=float(len(group))
		if size==0:
			continue
		
		p=np.count_nonzero(group==0,axis=0)
		if p[11]!=0:
			score=score-((p[11]/size)*math.log((p[11]/size),2))
		p=np.count_nonzero(group==1,axis=0)
		if p[11]!=0:
			score=score-((p[11]/size)*math.log((p[11]/size),2))
		p=np.count_nonzero(group==2,axis=0)
		if p[11]!=0:
			score=score-((p[11]/size)*math.log((p[11]/size),2))
		entropy=entropy+score*(size/totalinst)

	parentropy=0.0
	size=float(len(data))
	p=np.count_nonzero(data==0,axis=0)
	if p[11]!=0:
		parentropy=parentropy-((p[11]/size)*math.log((p[11]/size),2))
	p=np.count_nonzero(data==1,axis=0)
	if p[11]!=0:
		parentropy=parentropy-((p[11]/size)*math.log((p[11]/size),2))
	p=np.count_nonzero(data==2,axis=0)
	if p[11]!=0:
		parentropy=parentropy-((p[11]/size)*math.log((p[11]/size),2))
	return parentropy-entropy


#determine bestsplit attribute
def bestsplit(data,att_left):
	maxinfogain=-1000
	bestatr=-1
	b_split=[]
	for attr in att_left:
		grouping=splitnode(attr,data)
		info_gain=calcinfogain(data,grouping)
		if info_gain>maxinfogain:
			maxinfogain=info_gain
			bestatr=attr
			b_split=grouping
	return bestatr,b_split

#build the decision tree
def build_tree(node,data,att_unused):
	count=[]
	cur_count=np.count_nonzero(data==0,axis=0)
	count.append(cur_count[11])
	cur_count=np.count_nonzero(data==1,axis=0)
	count.append(cur_count[11])
	cur_count=np.count_nonzero(data==2,axis=0)
	count.append(cur_count[11])
	maxsize=max(count)
	if(len(data)!=0):
		node.majclass=count.index(max(count))

	if len(data)<10 or len(att_unused)==0 or maxsize==len(data):
		node.is_terminal=True
		if maxsize==len(data) and maxsize!=0:
			if maxsize==count[0]:
				node.terminal_class=0
			elif maxsize==count[1]:
				node.terminal_class=1
			elif maxsize==count[2]:
				node.terminal_class=2
		elif max(count)!=0 and len(data)<10 :
			node.terminal_class=count.index(max(count))
		else:
			node.terminal_class=node.parent.majclass
	else:
		bestatr,b_split_grp=bestsplit(data,att_unused)
		att_unused.remove(bestatr)
		node.attr=bestatr
		node.left=DT()
		node.left.parent=node
		node.left.parent_attribute=bestatr
		node.left.parent_attribute_val=0
		att_unused_left=[]
		for num in att_unused:
			att_unused_left.append(num);
		att_unused_leftmid=[]
		for num in att_unused:
			att_unused_leftmid.append(num);
		att_unused_rightmid=[]
		for num in att_unused:
			att_unused_rightmid.append(num);
		att_unused_right=[]
		for num in att_unused:
			att_unused_right.append(num);
		build_tree(node.left,b_split_grp[0],att_unused_left)
		node.leftmid=DT()
		node.leftmid.parent=node
		node.leftmid.parent_attribute=bestatr
		node.left.parent_attribute_val=1
		build_tree(node.leftmid,b_split_grp[1],att_unused_leftmid)
		node.rightmid=DT()
		node.rightmid.parent=node
		node.rightmid.parent_attribute=bestatr
		node.rightmid.parent_attribute_val=2
		build_tree(node.rightmid,b_split_grp[2],att_unused_rightmid)
		node.right=DT()
		node.right.parent=node
		node.right.parent_attribute=bestatr
		node.right.parent_attribute_val=3
		build_tree(node.right,b_split_grp[3],att_unused_right)
		node.num_children=4

#function to predict class label values on test set
def test_tree(node,data):
	if node.is_terminal==True:
		return node.terminal_class
	elif data[node.attr]==0:
		return test_tree(node.left,data)
	elif data[node.attr]==1:
		return test_tree(node.leftmid,data)
	elif data[node.attr]==2:
		return test_tree(node.rightmid,data)
	elif data[node.attr]==3:
		return test_tree(node.right,data)


#initilaising list
att_list=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide',
		'density','pH','sulphates','alcohol']

#shuffle data
np.random.shuffle(data)
#divide data into three sets

data1=np.zeros(shape=(533,12))
data1y=np.zeros(533)
data1=data[0:533,:]
data1y=data[0:533,11]

data2=np.zeros(shape=(533,12))
data2y=np.zeros(533)
data2=data[533:1066,:]
data2y=data[533:1066,11]

data3=np.zeros(shape=(533,12))
data3y=np.zeros(533)
data3=data[1066:1599,:]
data3y=data[1066:1599,11]

precision=0
accuracy=0
recall=0
accuracy_scikit=0
precision_scikit=0
recall_scikit=0
#decision tree using scikit learn
clf=DecisionTreeClassifier(criterion='entropy',min_samples_split=10)


train=np.zeros(shape=(1066,12))
test=np.zeros(shape=(533,12))
trainY=np.zeros(1066)
testpred=np.zeros(533)
testtrue=np.zeros(533)
train[0:533,:]=data1
train[533:1066,:]=data2
trainY=train[:,11]
test=data3
testtrue=data3y
#decision tree using scikit learn
clf.fit(train[:,0:11],trainY)
predictions=clf.predict(test[:,0:11])

#getting different measures
accuracy_scikit=accuracy_scikit+accuracy_score(testtrue,predictions)
precision_scikit=precision_scikit+precision_score(testtrue,predictions,average='macro')
recall_scikit=recall_scikit+recall_score(testtrue,predictions,average='macro')
#user defined decision tree

att_left=[0,1,2,3,4,5,6,7,8,9,10]
root1=DT()
root1.is_root=True
build_tree(root1,train,att_left)

i=0
for row in test:
	testpred[i]=test_tree(root1,row)
	i=i+1

accuracy=accuracy+accuracy_score(testtrue,testpred)
precision=precision+precision_score(testtrue,testpred,average='macro')
recall=recall+recall_score(testtrue,testpred,average='macro')

train=np.zeros(shape=(1066,12))
test=np.zeros(shape=(533,12))
testpred=np.zeros(533)
testtrue=np.zeros(533)
trainY=np.zeros(1066)
train[0:533,:]=data2
train[533:1066,:]=data3
trainY=train[:,11]
test=data1
testtrue=data1y

#decision tree using scikit learn
clf.fit(train[:,0:11],trainY)
predictions=clf.predict(test[:,0:11])
#getting different measures
accuracy_scikit=accuracy_scikit+accuracy_score(testtrue,predictions)
precision_scikit=precision_scikit+precision_score(testtrue,predictions,average='macro')
recall_scikit=recall_scikit+recall_score(testtrue,predictions,average='macro')
#user defined decision tree

att_left=[0,1,2,3,4,5,6,7,8,9,10]
root2=DT()
root2.is_root=True
build_tree(root2,train,att_left)


i=0
for row in test:
	testpred[i]=test_tree(root2,row)
	i=i+1



accuracy=accuracy+accuracy_score(testtrue,testpred)
precision=precision+precision_score(testtrue,testpred,average='macro')
recall=recall+recall_score(testtrue,testpred,average='macro')

train=np.zeros(shape=(1066,12))
test=np.zeros(shape=(533,12))
trainY=np.zeros(1066)
testpred=np.zeros(533)
testtrue=np.zeros(533)
train[0:533,:]=data3
train[533:1066,:]=data1
trainY=train[:,11]
test=data2
testtrue=data2y

#decision tree using scikit learn
clf.fit(train[:,0:11],trainY)
predictions=clf.predict(test[:,0:11])
#getting different measures
accuracy_scikit=accuracy_scikit+accuracy_score(testtrue,predictions)
precision_scikit=precision_scikit+precision_score(testtrue,predictions,average='macro')
recall_scikit=recall_scikit+recall_score(testtrue,predictions,average='macro')
#user defined decision tree
root3=DT()
root3.is_root=True
att_left=[0,1,2,3,4,5,6,7,8,9,10]
build_tree(root3,train,att_left)

i=0
for row in test:
	testpred[i]=test_tree(root3,row)
	i=i+1


accuracy=accuracy+accuracy_score(testtrue,testpred)
precision=precision+precision_score(testtrue,testpred,average='macro')
recall=recall+recall_score(testtrue,testpred,average='macro')
accuracy=accuracy/3
precision=precision/3
recall=recall/3
accuracy_scikit=accuracy_scikit/3
precision_scikit=precision_scikit/3
recall_scikit=recall_scikit/3

print('Mean Macro Accuracy :',accuracy)
print('Mean Macro Precision :',precision)
print('Mean Macro Recall :',recall)
print("Mean Macro Accuracy using scikit-learn :",accuracy_scikit)
print("Mean Macro Precision using scikit-learn :",precision_scikit)
print("Mean Macro Recall using scikit-learn :",recall_scikit)









