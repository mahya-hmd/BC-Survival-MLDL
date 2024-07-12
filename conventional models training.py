#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[2]:


data = pd.read_csv(r'path to training dataset')


X = data.drop(columns=['Survival status', 'ID', 'IORT'])  
y = data['Survival status']
print(X[0:5])
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
print('xxxxxxxxxxx')
print(X[0:5])


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[4]:


imputer = KNNImputer(n_neighbors=5)  
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)


# In[5]:


kfold = KFold(n_splits=5, shuffle=True, random_state=42)


# # KNN

# In[6]:


param_grid_knn = {
'n_neighbors': [3, 5, 7, 9,11, 25, 40, 62],  # Adjust the range as needed
'weights': ['uniform', 'distance'],
"metric" : ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
'algorithm' : ['ball_tree', 'kd_tree', 'brute', 'auto'],
'leaf_size' : [10, 20, 30, 40]
}

knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, cv=kfold)
grid_search_knn.fit(X_train_imputed, y_train)



# In[7]:


best_knn = grid_search_knn.best_estimator_  # For GridSearchCV
# best_knn = bayes_search.best_estimator_  # For BayesSearchCV

best_params_knn = grid_search_knn.best_params_  # For GridSearchCV
# best_params = bayes_search.best_params_  # For BayesSearchCV

model_data = {
    'model': best_knn,
    'hyperparameters': best_params_knn,
}
joblib.dump(model_data, 'path to save the model\model.pkl', protocol=4)


# In[8]:


y_train_pred = best_knn.predict(X_train_imputed)
y_test_pred = best_knn.predict(X_test_imputed)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Best KNN Model Parameters:", best_params_knn)
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")



# # DT

# In[9]:


# Defining hyperparameter
param_grid_dt = {
'max_depth' : [None, 5, 10, 20,30,40],
'min_samples_split' : [2, 5, 10, 15, 20],
'min_samples_leaf' : [1, 2, 4, 8, 16],
'criterion' : ["gini", "entropy"],
'random_state' : [42, 14, 27],
'max_features' : ['auto', 'sqrt', '1og2', 0.5]
    
}

decision_tree = DecisionTreeClassifier(random_state=42)
grid_search_dt = GridSearchCV(estimator=decision_tree, param_grid=param_grid_dt, cv=kfold)
grid_search_dt.fit(X_train_imputed, y_train)


# In[10]:


best_decision_tree = grid_search_dt.best_estimator_
best_params_dt = grid_search_dt.best_params_

model_data = {
    'model': best_decision_tree,
    'hyperparameters': best_params_dt,
}
joblib.dump(model_data, 'path to save the model\model.pkl', protocol=4)


# In[11]:


# Training Accuracy
y_train_pred = best_decision_tree.predict(X_train_imputed)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Test Accuracy
y_test_pred = best_decision_tree.predict(X_test_imputed)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Best Decision Tree Model Parameters:", best_params_dt)
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


# # RF

# In[6]:


param_grid_rf = {
'n_estimators': [10, 50, 100,200],
'max_depth': [None, 5, 10, 15, 20, 30],
'min_samples_split' : [2, 3, 5, 10, 20],
'min_samples_leaf' : [1,2,4],
'max_features' : ['auto', 'sqrt', 'log2', 0.5,3],
'criterion' : ["entropy"],
'random_state' : [42]
}




random_forest = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(estimator=random_forest, param_grid=param_grid_rf, cv=kfold)
grid_search_rf.fit(X_train_imputed, y_train)


# In[7]:


best_random_forest = grid_search_rf.best_estimator_
best_params_rf = grid_search_rf.best_params_

model_data = {
    'model': best_random_forest,
    'hyperparameters': best_params_rf,
}
joblib.dump(model_data, 'path to save the model\model.pkl', protocol=4)


# In[8]:


# Training Accuracy
y_train_pred = best_random_forest.predict(X_train_imputed)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Test Accuracy
y_test_pred = best_random_forest.predict(X_test_imputed)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Best Random Forest Model Parameters:", best_params_rf)
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


# # NB

# In[19]:


naive_bayes = GaussianNB()

# Define a parameter grid for hyperparameter tuning
param_grid_nb = {
    'priors': [None, [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]]
#         "force_alpha": [True, False],
#     "fit_prior": [True, False]
}

# Use GridSearchCV to find the best Naive Bayes model
grid_search_nb = GridSearchCV(naive_bayes, param_grid_nb, cv=kfold, scoring='accuracy')
grid_search_nb.fit(X_train_imputed, y_train)

# Get the best Naive Bayes model
best_naive_bayes = grid_search_nb.best_estimator_

# Train the best Naive Bayes model on the full training set
best_naive_bayes.fit(X_train_imputed, y_train)
best_params_nb = grid_search_nb.best_params_

# Make predictions on both the training and test sets
y_train_pred = best_naive_bayes.predict(X_train_imputed)
y_test_pred = best_naive_bayes.predict(X_test_imputed)

# Calculate accuracy on both the training and test datasets
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

print(best_params_nb)


# In[16]:


model_data = {
    'model': best_naive_bayes,
    'hyperparameters': best_params_nb,
}
joblib.dump(model_data, 'path to save the model\model.pkl', protocol=4)


# # Extra Trees

# In[6]:


param_grid_ext = {
'n_estimators' : [200, 300],  
'max_depth' : [None, 5, 10, 15, 20, 30] ,
'min_samples_split' : [2, 5, 15, 20],
'min_samples_leaf' : [1, 2, 4, 8, 16]  ,
'max_features' : ['auto', 'sqrt', 'log2', 0.5,1],
'random_state' : [27]
}




extra_trees = ExtraTreesClassifier(random_state=42)
grid_search_ext = GridSearchCV(estimator=extra_trees, param_grid=param_grid_ext, cv=kfold)
grid_search_ext.fit(X_train_imputed, y_train)



# In[7]:


best_extra_trees = grid_search_ext.best_estimator_
best_params_ext = grid_search_ext.best_params_

model_data = {
    'model': best_extra_trees,
    'hyperparameters': best_params_ext,
}
joblib.dump(model_data, 'path to save the model\model.pkl', protocol=4)


# In[8]:


# Training Accuracy
y_train_pred = best_extra_trees.predict(X_train_imputed)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Test Accuracy
y_test_pred = best_extra_trees.predict(X_test_imputed)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Best Extra Trees Model Parameters:", best_params_ext)
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


# # GBoost

# In[6]:


gradient_boosting = GradientBoostingClassifier()

# Defining hyperparameter
param_grid_gb = {
'n_estimators': [100, 200, 300, 400],
'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.4],
'max_depth': [3, 4, 5, 6],
'min_samples_split': [2, 5, 10, 15],
'min_samples_leaf': [1, 2, 4],
'subsample': [0.8, 0.9, 1.0]
}

grid_search_gb = GridSearchCV(estimator=gradient_boosting, param_grid=param_grid_gb, scoring='accuracy', cv=kfold)
grid_search_gb.fit(X_train_imputed, y_train)


# In[7]:


best_gradient_boosting = grid_search_gb.best_estimator_
best_params_gb = grid_search_gb.best_params_

model_data = {
    'model': best_gradient_boosting,
    'hyperparameters': best_params_gb,
}
joblib.dump(model_data, 'path to save the model\model.pkl', protocol=4)


# In[8]:


# Training Accuracy
y_train_pred = best_gradient_boosting.predict(X_train_imputed)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Test Accuracy
y_test_pred = best_gradient_boosting.predict(X_test_imputed)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Best Gradient Boosting Model Parameters:", best_params_gb)
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


# # AdaBoost

# In[9]:


base_estimator = DecisionTreeClassifier(max_depth=1)  # You can adjust the base estimator

ada_boost = AdaBoostClassifier(base_estimator=base_estimator)

# Defining hyperparameter
param_grid_ada = {
    'n_estimators': [50, 100, 200],  # Adjust the number of estimators as needed
    'learning_rate': [0.01, 0.1, 1.0], # Adjust the learning rate as needed
    'base_estimator' : [DecisionTreeClassifier(max_depth=1),DecisionTreeClassifier(max_depth=2)]
}

grid_search_ada = GridSearchCV(estimator=ada_boost, param_grid=param_grid_ada, scoring='accuracy', cv=kfold)
grid_search_ada.fit(X_train_imputed, y_train)


# In[10]:


best_ada_boost_model = grid_search_ada.best_estimator_
best_params_ada = grid_search_ada.best_params_

model_data = {
    'model': best_ada_boost_model,
    'hyperparameters': best_params_ada,
}
joblib.dump(model_data, 'path to save the model\model.pkl', protocol=4)


# In[11]:


# Training Accuracy
y_train_pred = best_ada_boost_model.predict(X_train_imputed)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Test Accuracy
y_test_pred = best_ada_boost_model.predict(X_test_imputed)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Best AdaBoost Model Parameters:", best_params_ada)
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


# # XGBoost

# In[12]:


xgb_classifier = xgb.XGBClassifier()

# Defining hyperparameter
param_grid_xgb = {
'n_estimators' : [50, 100, 200, 300, 400],
'learning_rate' : [0.01, 0.1, 0.3, 0.4],
'max_depth' : [3, 4, 5, 6],
'alpha' : [0, 1, 2],
'min_child_weight' : [1, 2, 4, 8],
'subsample' : [0.8, 0.9, 1.0],
'colsample_bytree' : [0.8, 0.9, 1.0],
'early_stopping'  : [True]
}

grid_search_xgb = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid_xgb, scoring='accuracy', cv=kfold)
grid_search_xgb.fit(X_train_imputed, y_train)


# In[13]:


best_xgb_model = grid_search_xgb.best_estimator_
best_params_xgb = grid_search_xgb.best_params_

model_data = {
    'model': best_xgb_model,
    'hyperparameters': best_params_xgb,
}
joblib.dump(model_data, 'path to save the model\model.pkl', protocol=4)


# In[14]:


# Training Accuracy
y_train_pred = best_xgb_model.predict(X_train_imputed)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Test Accuracy
y_test_pred = best_xgb_model.predict(X_test_imputed)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Best XGBoost Model Parameters:", best_params_xgb)
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


# # LR

# In[12]:


logistic_regression = LogisticRegression()

# Defining hyperparameter
param_grid_lr = {
'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'newton-cholesky', 'sag', 'saga'] , 
'max_iter' : [100, 500, 800, 1000],          
'class_weight' : [None, 'balanced']  ,
'random_state' : [27, 42]
}

grid_search_lr = GridSearchCV(estimator=logistic_regression, param_grid=param_grid_lr, scoring='accuracy', cv=kfold)
grid_search_lr.fit(X_train_imputed, y_train)


# In[13]:


best_logistic_regression = grid_search_lr.best_estimator_
best_params_lr = grid_search_lr.best_params_

model_data = {
    'model': best_logistic_regression,
    'hyperparameters': best_params_lr,
}
joblib.dump(model_data, 'path to save the model\model.pkl', protocol=4)


# In[14]:


# Training Accuracy
y_train_pred = best_logistic_regression.predict(X_train_imputed)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Test Accuracy
y_test_pred = best_logistic_regression.predict(X_test_imputed)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Best Logistic Regression Model Parameters:", best_params_lr)
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


# # SVM

# In[6]:


svm_classifier = SVC()

# Defining hyperparameter
param_grid_svm = {
'C': [0.001, 1],  
'kernel': ['rbf'],  
'gamma': [0.001, 1], 
'max_iter' : [200, 300, 500],
'degree' : [2, 3, 4],
'class_weight' : [None, 'balanced'],
'probability': [True]
    
}





grid_search_svm = GridSearchCV(estimator=svm_classifier, param_grid=param_grid_svm, scoring='accuracy', cv=kfold)
grid_search_svm.fit(X_train_imputed, y_train)


# In[7]:


best_svm_model = grid_search_svm.best_estimator_
best_params_svm = grid_search_svm.best_params_

model_data = {
    'model': best_svm_model,
    'hyperparameters': best_params_svm,
}
joblib.dump(model_data, 'path to save the model\model.h5')


# In[8]:


# Training Accuracy
y_train_pred = best_svm_model.predict(X_train_imputed)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Test Accuracy
y_test_pred = best_svm_model.predict(X_test_imputed)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Best SVM Model Parameters:", best_params_svm)
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


# # MLP

# In[6]:


param_grid_MLP = {

    "activation": ["identity", "logistic", "tanh", "relu"],
    "learning_rate": ['constant', 'invscaling', 'adaptive'],
    'learning_rate_init': [0.001, 0.01, 0.1],
    "solver": ['adam', 'lbfgs', 'sgd'],
    "max_iter": [2000, 3000, 5000],
    'random_state': [42,27,14]

}




# Define the MLPClassifier with increased max_iter
mlp_classifier = MLPClassifier()
grid_search_MLP = GridSearchCV(mlp_classifier, param_grid_MLP, cv=kfold, scoring='accuracy', error_score='raise')

grid_search_MLP.fit(X_train_imputed, y_train)


# In[8]:


# Get the best MLP model from the search
best_mlp_model = grid_search_MLP.best_estimator_
best_params = grid_search_MLP.best_params_

# Training Accuracy
y_train_pred = best_mlp_model.predict(X_train_imputed)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Test Accuracy
y_test_pred = best_mlp_model.predict(X_test_imputed)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Evaluate the selected model on the test dataset
#test_accuracy = best_mlp_model.score(X_test, y_test)

print("Best MLP Model Hyperparameters:", grid_search_MLP.best_params_)
print("Training Accuracy:", train_accuracy)
print("Test Accuracy with Best MLP Model:", test_accuracy)


# In[9]:


model_data = {
    'model': best_mlp_model,
    'hyperparameters': best_params,
}
joblib.dump(model_data, 'path to save the model\model.pkl')


# # DNN

# In[ ]:





# In[ ]:





# In[ ]:




