from prediction import predict
import streamlit as st
import numpy as np

st.title("Iris Species Prediction")
st.write("""
This tool allows you to predict the species of an Iris flower based on its characteristics.
Please enter the dimensions of the sepals and petals below:
""")

# Split the interface into two columns
col1, col2 = st.columns(2)

# Flower characteristics input with sliders
with col1:
    sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.1)
    sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.5)

with col2:
    petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 1.4)
    petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 0.2)


# Button to make the prediction
if st.button('Predict Species'):
    features = [sepal_length, sepal_width, petal_length, petal_width]
    prediction = predict(features)
    st.write(f"Predicted Species: **{prediction}**")
