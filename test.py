
import pandas as pd
from matplotlib import pyplot as plt
test=pd.read_csv('test.csv')
X_train=test
'''
g = sns.countplot(x=Y_train)
print(Y_train.value_counts())  #数据的可视化
plt.show()    
'''
x=X_train.iloc[1].values.reshape(28,28) 
g= plt.imshow(x,cmap=plt.cm.gray_r)     
plt.show()           
'''
import pandas as pd
test=pd.read_csv('test.csv')
print(len(test))
'''

#这个文件没啥用