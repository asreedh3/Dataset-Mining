

# Importing the dataset
from sklearn import datasets
mydata=datasets.load_diabetes()
X = mydata.data
y = mydata.target

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)

# Fitting the Regression tree Model to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 100)

# Applying grid search to find the best model and the best model parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'criterion': ["mse", "friedman_mse", "mae"], 'splitter': ["best", "random"], 'max_depth': [1, 2, 3], 'min_samples_split': [2, 3, 4]}]
grid_search = GridSearchCV(estimator = regressor, param_grid = parameters, scoring = 'r2', cv = 10)
if __name__ == '__main__':
    grid_search = grid_search.fit(X_train, y_train)
print ("The accuracy of the training set:",grid_search.best_score_)
print ("The best parameters are:",grid_search.best_params_)
best_model = grid_search.best_estimator_
yhat = best_model.predict(X_test)
# Obtaining accuracy score for the test set
from sklearn.metrics import r2_score
print ("The accuracy for the test set is:",r2_score(y_test,yhat))

# Visualization
from sklearn import tree
import graphviz
dot_data = tree.export_graphviz(best_model,out_file=None)
graph = graphviz.Source(dot_data)
graph.render("TreeModelExample1")
graph.view()