import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
import time
train=pd.read_csv('train.csv')
Y_train = train["label"]                                     #把标签单独整出来
X_train = train.drop(labels = ["label"],axis = 1)                #把剩下的数据提出来   
del train                                                       #没用了
Y_train=Y_train.values.reshape(42000,1)                         #有点烂
X_train =X_train.values.reshape(42000, 784).astype('float32')   #转一下格式
X_train = X_train / 255.0                                       #归一化
x_train,x_test,y_train,y_test=train_test_split(X_train,Y_train) #浅浅分一下数据集
clf = svm.SVC(C=100.0, kernel='rbf', gamma=0.03)                #随便调一下
t1 = time.time()                            
clf.fit(x_train, y_train)                   
t2 = time.time()
ustime = float(t2-t1)
print("训练时间:{}s".format(ustime))
predictions = [int(a) for a in clf.predict(x_test)]             
print('准确率', accuracy_score(y_test, predictions))


