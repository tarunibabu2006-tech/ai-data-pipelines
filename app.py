
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: 'model.pkl' not found. Please ensure the model file is in the same directory.")
    st.stop()

st.title('Car MPG Prediction App')

st.write('Enter the car features to predict its Miles Per Gallon (MPG).')

# Input features (based on the model trained: 'horsepower', 'weight')
# These ranges are based on the 'auto-mpg' dataset's descriptive statistics.
horsepower = st.slider('Horsepower', min_value=40, max_value=250, value=100, step=5)
weight = st.slider('Weight (lbs)', min_value=1500, max_value=5500, value=2500, step=50)

if st.button('Predict MPG'):
    # Prepare the input for the model
    input_data = pd.DataFrame([[horsepower, weight]], columns=['horsepower', 'weight'])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    st.success(f'Predicted MPG: {prediction:.2f}')

st.write('---')
st.write('This app uses a Linear Regression model trained on car features to predict MPG.')
