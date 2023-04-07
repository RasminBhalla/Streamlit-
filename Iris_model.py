# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 10:34:11 2023

@author: RasminBhalla
"""

import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)


import pickle
pickle.dump(clf, open('iris_clf.pkl', 'wb'))


# =============================================================================
# 
# prediction = clf.predict(df)
# prediction_proba = clf.predict_proba(df)
# 
# =============================================================================
