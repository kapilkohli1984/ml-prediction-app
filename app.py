
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load models
with open("lr_model.pkl", "rb") as f:
    lr_model = pickle.load(f)
with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

st.set_page_config(page_title="Ultimate ML App", layout="wide")
st.title("🚀 Ultimate ML Prediction App")
st.write("Predict target values, compare models, and explore feature importance.")

# Single Prediction
st.header("Single Prediction")
model_choice = st.selectbox("Choose a model:", ["Linear Regression", "Random Forest"])
input_value = st.number_input("Enter a value for prediction:", value=0.0, step=0.1)

if st.button("Predict"):
    model = lr_model if model_choice == "Linear Regression" else rf_model
    prediction = model.predict(np.array(input_value).reshape(-1, 1))
    st.metric("Prediction", f"{prediction[0]:.2f}")
