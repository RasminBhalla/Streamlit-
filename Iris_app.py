# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 10:43:04 2023

@author: RasminBhalla
"""

import streamlit as st
import pandas as pd
import pickle
from sklearn import datasets
st.write("""
#  Iris Flower Prediction App
* This app predicts the **Iris flower** type!
""")

st.sidebar.header('User Input Parameters')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
        sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
        petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
        petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)
        data = {'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width}
        features = pd.DataFrame(data, index=[0])
        return features

iris = datasets.load_iris()
X = iris.data
Y = iris.target
df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

load_clf = pickle.load(open('iris_clf.pkl', 'rb'))

prediction = load_clf.predict(df)
prediction_proba =load_clf.predict_proba(df)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)

