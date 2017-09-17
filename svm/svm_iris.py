
"""
Created on Fri Sep 15 14:25:43 2017

@author: appel
"""
import pandas as pd
data = pd.read_csv("DataSet/iris.data.txt")
params = data.iloc[:,:-1].values
iris = data.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
labelEncoder_1 = LabelEncoder()
iris = labelEncoder_1.fit_transform(iris)

from sklearn.model_selection import train_test_split
train_data,test_data,train_target,test_target=train_test_split(params,iris,test_size=0.2)

from sklearn import svm
clf = svm.SVC()
clf.fit(train_data,train_target)
clf.support_
clf.support_vectors_
clf.n_support_
pred_target = clf.predict(test_data)

from sklearn.metrics import confusion_matrix

res = confusion_matrix(test_target,pred_target)
print("accuracy:",float(res[0,0]+res[1,1]+res[2,2])/len(test_target))
