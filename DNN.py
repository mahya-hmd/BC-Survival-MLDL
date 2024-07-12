#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import autokeras as ak
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import joblib
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow import keras
from sklearn.feature_selection import f_classif, SelectKBest



# In[3]:


data_set_train = pd.read_csv(r'path to train dataset')
data_set_test = pd.read_csv(r'path to test dataset')

X_train = data_set_train.drop(columns=['Survival status', 'All the unwanted features'])  
y_train = data_set_train['Survival status']

X_test = data_set_test.drop(columns=['Survival status', 'All the unwanted features'])  
y_test = data_set_test['Survival status']


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# imputation
imputer = KNNImputer(n_neighbors=5)  
X_train = imputer.fit_transform(X_train)
X_test = imputer.fit_transform(X_test)



# Spliting - shiraz
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train,  
    y_train,   
    test_size=0.2,               
    random_state=42
)

clf = ak.StructuredDataClassifier(max_trials=100)  # Adjust max_trials as needed
clf.fit(X_train, y_train, epochs=10, validation_data=(X_valid,y_valid))
best_model_ak = clf.export_model()


# In[4]:


print(clf.tuner.search_space_summary())


# In[8]:


# Train data
train_predictions = best_model_ak.predict(X_train)
test_predictions = best_model_ak.predict(X_test)

# Calculate metrics
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

threshold = 0.5
train_predictions_binary = (train_predictions > threshold).astype(int)
test_predictions_binary = (test_predictions > threshold).astype(int)

# # Validation data
validation_predictions = best_model_ak.predict(X_valid)  # Replace x_validation with your validation data

# # Calculate validation metrics
validation_accuracy = accuracy_score(y_valid, (validation_predictions > threshold).astype(int))
validation_auc = roc_auc_score(y_valid, validation_predictions)
validation_confusion = confusion_matrix(y_valid, (validation_predictions > threshold).astype(int))
validation_specificity = validation_confusion[0, 0] / (validation_confusion[0, 0] + validation_confusion[0, 1])
validation_sensitivity = validation_confusion[1, 1] / (validation_confusion[1, 0] + validation_confusion[1, 1])



# # # Train metrics
train_accuracy = accuracy_score(y_train, train_predictions_binary)
# train_auc = roc_auc_score(y_train, train_predictions)
# train_confusion = confusion_matrix(y_train, (train_predictions > 0.5).astype(int))
# train_specificity = train_confusion[0, 0] / (train_confusion[0, 0] + train_confusion[0, 1])
# train_sensitivity = train_confusion[1, 1] / (train_confusion[1, 0] + train_confusion[1, 1])

# # # Test metrics
test_accuracy = accuracy_score(y_test, test_predictions_binary)
test_auc = roc_auc_score(y_test, test_predictions)
test_confusion = confusion_matrix(y_test, (test_predictions > 0.5).astype(int))
test_specificity = test_confusion[0, 0] / (test_confusion[0, 0] + test_confusion[0, 1])
test_sensitivity = test_confusion[1, 1] / (test_confusion[1, 0] + test_confusion[1, 1])


# In[9]:


print("train_accuracy : ", train_accuracy)
print("validation_accuracy : ", validation_accuracy)
print("validation_auc : ", validation_auc)
print("validation_specificity : ", validation_specificity)
print("validation_sensitivity : ", validation_sensitivity)


print("test_accuracy : ", test_accuracy)
print("test_auc : ", test_auc)
print("test_specificity : ", test_specificity)
print("test_sensitivity : ", test_sensitivity)


# In[ ]:




