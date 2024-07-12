#!/usr/bin/env python
# coding: utf-8

# In[23]:


from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import KNNImputer
from sklearn.feature_selection import f_classif, SelectKBest



# In[24]:


data = pd.read_csv(r'C:\lab\py\out\masterdata\BRC-SH3.csv')


X = data.drop(columns=['Survival status', 'ID', 'IORT'])  
y = data['Survival status']


# Step 1: Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Imputation Using KNN
knn_imputer = KNNImputer()
X_train_imputed = knn_imputer.fit_transform(X_train)
X_test_imputed = knn_imputer.transform(X_test)

kf = KFold(n_splits=5, shuffle=True, random_state=42)


# In[25]:


k_best = SelectKBest(score_func=f_classif, k='all')  # You can change k to the number of features you want to select

# Fit and transform the data
X_selected = k_best.fit_transform(X_train_imputed, y_train)

# Get the selected features
selected_features = X.columns[k_best.get_support()]

# Print the selected features
print("Selected Features:", selected_features)


p_values = k_best.pvalues_
print("P-values for Selected Features:")
for feature, p_value in zip(selected_features, p_values):
    print(f"{feature}: {p_value:.4f}")


    
scaler = MinMaxScaler()
X_train_imputed = scaler.fit_transform(X_train_imputed)
X_test_imputed = scaler.fit_transform(X_test_imputed)
X_train_selected = scaler.fit_transform(X_selected)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




