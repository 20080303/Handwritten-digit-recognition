import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
import numpy as np
import time
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
Y_train = train["label"]                                     #把标签单独整出来
X_train = train.drop(labels = ["label"],axis = 1)                #把剩下的数据提出来   
del train                                                       #没用了
Y_train=Y_train.values.reshape(42000,1)                         #有点烂
X_train =X_train.values.reshape(42000, 784).astype('float32')   #转一下格式
X_train = X_train / 255.0  #归一化
long=len(test)
Test=test.values.reshape(long,784)
Test = Test / 255.0  #归一化
del test 
clf = svm.SVC(C=100.0, kernel='rbf', gamma=0.03)                #随便调一下
t1 = time.time()                            
clf.fit(X_train, Y_train)                   
predictions = [int(a) for a in clf.predict(Test)]             #搞成数组比较好放csv
i=np.arange(long)
dict = {'ID': i, 'pre': predictions}
result = pd.DataFrame(dict)
result.to_csv('result.csv')