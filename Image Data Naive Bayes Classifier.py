# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 19:44:57 2018

@author: Ashlin
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from pandas_ml import ConfusionMatrix
from sklearn import model_selection
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
mydata = pd.read_csv("Imagedata.csv")
target = mydata["Image_Classification"] 
X=mydata.iloc[:,1:19]

y=target
modelnow=GaussianNB()
modelnow.fit(X,y)

yhat = modelnow.predict(X)
print "Training Error"
print metrics.accuracy_score(y, yhat)
print metrics.classification_report(y, yhat)
print ConfusionMatrix(y, yhat)
seed=5000
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

modelnow=GaussianNB()
modelnow.fit(X_train,y_train)

yhat = modelnow.predict(X_test)
print "Testing Error"
print metrics.accuracy_score(y_test, yhat)
print metrics.classification_report(y_test, yhat)
print ConfusionMatrix(y_test, yhat)


actuals=[]
probs=[]
hats=[]
target = mydata["Image_Classification"] 
X=mydata.iloc[:,1:19]

y=target
seed=9000
kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)
for train, test in kfold.split(X, y):
    
    
    
   
    modelnow.fit(X.iloc[train],y.iloc[train])
    foldhats = modelnow.predict(X.iloc[test])
    foldprobs = modelnow.predict_proba(X.iloc[test])[:,0] 
    actuals = np.append(actuals, y.iloc[test])
    probs = np.append(probs, foldprobs)
    hats = np.append(hats, foldhats)
    
print "Crossvalidation Error"    
print "CVerror = ", metrics.accuracy_score(actuals,hats)
print metrics.classification_report(actuals, hats)
cm = ConfusionMatrix(actuals,hats)

cm.print_stats()

hats1=pd.get_dummies(hats)
actuals1=pd.get_dummies(actuals)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(0,7,1):
    fpr[i], tpr[i], _ = roc_curve(actuals1.iloc[:,i], hats1.iloc[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

print("ROC Curve Class: BRICKFACE")
plt.figure(figsize=(6,6))
lw = 3
plt.plot(fpr[0], tpr[0], color='red',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

print("ROC Curve Class: CEMENT")
plt.figure(figsize=(6,6))
lw = 3
plt.plot(fpr[1], tpr[1], color='red',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

print("ROC Curve Class: FOLIAGE")
plt.figure(figsize=(6,6))
lw = 3
plt.plot(fpr[2], tpr[2], color='red',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

print("ROC Curve Class: GRASS")
plt.figure(figsize=(6,6))
lw = 3
plt.plot(fpr[3], tpr[3], color='red',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[3])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

print("ROC Curve Class: PATH")
plt.figure(figsize=(6,6))
lw = 3
plt.plot(fpr[4], tpr[4], color='red',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[4])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

print("ROC Curve Class: SKY")
plt.figure(figsize=(6,6))
lw = 3
plt.plot(fpr[5], tpr[5], color='red',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[5])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

print("ROC Curve Class: WINDOW")
plt.figure(figsize=(6,6))
lw = 3
plt.plot(fpr[6], tpr[6], color='red',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[6])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()








