# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 11:54:04 2018

@author: Ashlin
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from pandas_ml import ConfusionMatrix

mydata=pd.read_csv('diabetes.csv')

X=mydata.iloc[:,0:8]

y=mydata.iloc[:,8]
seed=5000

scaler = StandardScaler()
scaler.fit(X)
Xstd = scaler.transform(X)
X=Xstd
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=seed)
print("Linear SVM")
svc = SVC(kernel='linear')
svc_grid = GridSearchCV(svc, {'C':[0.01,0.1,1,10,100,1000]},return_train_score=True, n_jobs=4, cv=5)
svc_grid.fit(X_train, y_train)
svc_best=svc_grid.best_estimator_ 
yhat=svc_best.predict(X_test)
print ConfusionMatrix(y_test,yhat)

print 'Best parameters are:',svc_grid.best_params_
print("The test accuracy is "),svc_best.score(X_test,y_test)
print("The training accuracy is"), svc_best.score(X_train,y_train)
print("Non Linear SVM")
svc = SVC()
svc_grid = GridSearchCV(svc, {'gamma':[0.00001,0.001,0.01,0.1,1,10,100], 'C':[0.01,0.1,1,10,100,1000],'kernel':['rbf','sigmoid']},return_train_score=True, n_jobs=4,cv=5)
svc_grid.fit(X_train, y_train)
svc_best=svc_grid.best_estimator_ 
yhat=svc_best.predict(X_test)
print ConfusionMatrix(y_test,yhat)

print 'Best parameters are:',svc_grid.best_params_
print("The test accuracy is "),svc_best.score(X_test,y_test)
print("The training accuracy is"), svc_best.score(X_train,y_train)