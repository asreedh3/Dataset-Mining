# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 21:07:52 2018

@author: Ashlin
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
mydata=pd.read_csv('diabetes.csv')

X=mydata.iloc[:,0:8]

y=mydata.iloc[:,8]

seed = 5000
scaler = StandardScaler()
scaler.fit(X)
Xstd = scaler.transform(X)
X=Xstd
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
print("Decision Tree")
parameters = {'max_depth':range(1,5),'min_samples_split':range(2,11)}
modelnow=GridSearchCV(DecisionTreeClassifier(), parameters, n_jobs=4, cv=5)
modelnow.fit(X_train,y_train)
best_model = modelnow.best_estimator_
print 'Best parameters are:',modelnow.best_params_
print("The test accuracy is "),best_model.score(X_test,y_test)
print("The training accuracy is"), best_model.score(X_train,y_train)

import graphviz
dot_data = tree.export_graphviz(best_model,out_file=None)
graph = graphviz.Source(dot_data)
graph.render("TreeModelExample")
graph.view()
print("Linear SVM")
svc = SVC(kernel='linear',random_state=seed)
svc_grid = GridSearchCV(svc, {'C':[0.01,0.1,1,10,100,1000]},return_train_score=True, n_jobs=4, cv=5)
svc_grid.fit(X_train, y_train)
svc_best=svc_grid.best_estimator_ 

print 'Best parameters are:',svc_grid.best_params_
print("The test accuracy is "),svc_best.score(X_test,y_test)
print("The training accuracy is"), svc_best.score(X_train,y_train)
print("Non Linear SVM")
svc = SVC(random_state=seed)
svc_grid = GridSearchCV(svc, {'gamma':[0.00001,0.001,0.01,0.1,1,10,100], 'C':[0.01,0.1,1,10,100,1000]},return_train_score=True, n_jobs=4,cv=5)
svc_grid.fit(X_train, y_train)
svc_best=svc_grid.best_estimator_ 

print 'Best parameters are:',svc_grid.best_params_
print("The test accuracy is "),svc_best.score(X_test,y_test)
print("The training accuracy is"), svc_best.score(X_train,y_train)
print("Random Forest")
parameters={'n_estimators': [100,500,1000], 'max_features': ['sqrt','log2','auto'],'random_state':[seed],'oob_score':[True]}
modelnow=RandomForestClassifier()
rf_grid=GridSearchCV(modelnow,parameters,cv=5,n_jobs =4)
rf_grid.fit(X_train,y_train)
rf_best=rf_grid.best_estimator_
print("Best Parameters"),rf_grid.best_params_
print("Training Score"),rf_best.score(X_train,y_train)
print("Testing Score"),rf_best.score(X_test,y_test)
print("AdaBoost")
parameters={'n_estimators':[50,100,500,1000]}
ada=AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
ada_grid=GridSearchCV(ada,parameters,n_jobs=4,cv=5)
ada_grid.fit(X_train,y_train)
ada_best=ada_grid.best_estimator_
print("Best Parameters"),ada_grid.best_params_
print("Training Score"),ada_best.score(X_train,y_train)
print("Testing Score"),ada_best.score(X_test,y_test)
parameters={'n_estimators':[50,100,500,100],'learning_rate':[0.0001,0.001,0.01,0.05,0.1,0.2,0.3],'max_depth':range(2,10)}
xg=XGBClassifier(base_estimator=DecisionTreeClassifier())
xg_grid=GridSearchCV(xg,parameters,n_jobs=4,cv=5)
xg_grid.fit(X_train,y_train)
xg_best=xg_grid.best_estimator_
print("Best Parameters"),xg_grid.best_params_
print("Training Score"),xg_best.score(X_train,y_train)
print("Testing Score"),xg_best.score(X_test,y_test)

