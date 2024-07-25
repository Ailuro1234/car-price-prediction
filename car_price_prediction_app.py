import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.write("""
# Car Selling Price Prediction App

This app predicts the **selling price** of a car!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    year = st.sidebar.slider('Year', 2000, 2024, 2015)
    kms_driven = st.sidebar.slider('Kms Driven', 0, 500000, 30000)
    fuel_type = st.sidebar.selectbox('Fuel Type', ('Petrol', 'Diesel'))
    seller_type = st.sidebar.selectbox('Seller Type', ('Dealer', 'Individual'))
    transmission = st.sidebar.selectbox('Transmission', ('Manual', 'Automatic'))

    fuel_type_diesel = 1 if fuel_type == 'Diesel' else 0
    fuel_type_petrol = 1 if fuel_type == 'Petrol' else 0
    seller_type_individual = 1 if seller_type == 'Individual' else 0
    transmission_manual = 1 if transmission == 'Manual' else 0

    data = {'Year': year,
            'Kms_Driven': kms_driven,
            'Fuel_Type_Diesel': fuel_type_diesel,
            'Fuel_Type_Petrol': fuel_type_petrol,
            'Seller_Type_Individual': seller_type_individual,
            'Transmission_Manual': transmission_manual}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

prediction = model.predict(df)

st.subheader('Prediction')
st.write(f'The predicted selling price of the car is: {prediction[0]:.2f} lakh')

