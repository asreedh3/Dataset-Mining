# Decision Tree Classifier

# Importing the libraries

import pandas as pd
from pandas_ml import ConfusionMatrix

# Importing the dataset
dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:, : -1].values
y = dataset.iloc[:, 8].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 5000)

# Fitting classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 5000)

# Applying grid search to find the best model and the best model parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'criterion': ["gini", "entropy"], 'splitter': ["best", "random"], 'max_depth': [1, 2, 3], 'min_samples_split': [2, 3, 4]}]
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 5)
if __name__ == '__main__':
    grid_search = grid_search.fit(X_train, y_train)
print ("the training accuracy is:",grid_search.best_score_) 
print ("The best parameters are:",grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Confusion matrix for the best_model

print ConfusionMatrix(y_test, y_pred)

#The accuracy of the test set:
from sklearn.metrics import accuracy_score
print ("The test set accuracy is:",accuracy_score(y_test, y_pred))

# Visualization
from sklearn import tree
import graphviz
dot_data = tree.export_graphviz(best_model,out_file=None)
graph = graphviz.Source(dot_data)
graph.render("TreeModelExample1")
graph.view()