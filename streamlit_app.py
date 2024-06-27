import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('iris_model.pkl')

# Define the class names
class_names = ['Setosa', 'Versicolor', 'Virginica']

# Streamlit UI
st.title("Iris Flower Classification")

st.sidebar.header("Input Features")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input Features')
st.write(df)

# Make predictions
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

st.subheader('Prediction')
st.write(class_names[prediction[0]])

st.subheader('Prediction Probability')
st.write(prediction_proba)
