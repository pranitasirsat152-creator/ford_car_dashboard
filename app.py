
import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load("LR_model(1).pkl")
scaler = joblib.load("scaler(1).pkl")
model_columns = joblib.load("columns(1).pkl")

st.title("Car Price Prediction App")

model_name = st.selectbox("Model",["Fiesta","Focus","EcoSport","Kuga","Mondeo"])
year = st.number_input("Year",2000,2025,2020)
mileage = st.number_input("Mileage",0,200000,50000)
tax = st.number_input("Tax",0,500,150)
mpg = st.number_input("MPG",10.0,100.0,50.0)
engineSize = st.number_input("Engine Size",1.0,5.0,1.5)
transmission = st.selectbox("Transmission",["Manual","Automatic","Semi-Auto"])
fuelType = st.selectbox("Fuel Type",["Petrol","Diesel","Hybrid","Electric"])

if st.button("Predict Price"):
    input_data = pd.DataFrame([{
        "model":model_name,
        "year":year,
        "mileage":mileage,
        "tax":tax,
        "mpg":mpg,
        "engineSize":engineSize,
        "transmission":transmission,
        "fuelType":fuelType
    }])

    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=model_columns,fill_value=0)
    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)

    st.success(f"Predicted Price: £ {prediction[0]:.2f}")
