# Xgboost_model
give prediction based on category to the customers
import csv
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn import cross_validation as cv
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score
data = pd.read_csv('F:/finalfile.csv', encoding='latin-1')
nonnumeric_columns = ['name', 'Last Name', 'Gender', 'Age','event', 'group', 'time', 'city', 'state', 'url']
le = LabelEncoder()
for feature in nonnumeric_columns:
    data[feature] = le.fit_transform(data[feature])
X = data.iloc[:,0:13]
y = data.iloc[:,13]
gbc = GradientBoostingClassifier(max_depth=1,n_estimators = 404, warm_start = True, random_state = 80)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state =80)
gbc.fit(X_train,y_train)
gbc_y_pred = gbc.predict(X_test)
gbc_y_pred_prob = gbc.predict_proba(X_test)
gbc_accuracy =accuracy_score(y_test, gbc_y_pred)
gbc_logloss = log_loss(y_test, gbc_y_pred_prob)
print ("== gradient boosting ==")
print("Accuracy: {0:.2f}".format (gbc_accuracy))
print("logloss:{0:.2f}".format(gbc_logloss))
#print("Precision: {0:.2f}".format (precision_score(y_test, weighted_prediction, average='micro')))
#print("Recall: {0:.2f}".format (recall_score(y_test, weighted_prediction, average='micro')))
