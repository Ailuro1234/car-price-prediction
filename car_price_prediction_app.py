import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained models and scaler
with open('linear_regression_model.pkl', 'rb') as f:
    linear_model = pickle.load(f)

with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Set the title and sidebar header
st.set_page_config(page_title="Car Price Prediction App", layout="centered")

# Define pages
def welcome_page():
    st.title("🚗 Welcome to the Car Selling Price Prediction App")
    st.markdown("""
    This app predicts the **selling price** of a car!  
    Navigate through the app to get your car's estimated selling price.
    """)
    if st.button('Start'):
        st.session_state.page = 'main'

def main_page():
    st.title("🔧 Car Selling Price Prediction")
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

    # Scale the input features
    df_scaled = scaler.transform(df)

    st.subheader('📋 User Input Parameters')
    st.write(df)

    # Select model
    model_choice = st.sidebar.selectbox('Select Model', ('Linear Regression', 'Random Forest'))

    # Predict the car price
    if model_choice == 'Linear Regression':
        prediction = linear_model.predict(df_scaled)
    else:
        prediction = rf_model.predict(df_scaled)

    # Handle negative predictions
    if prediction[0] < 0:
        st.subheader('💰 Prediction')
        st.write(f'The predicted selling price of the car is too low to be realistic. Please check your input parameters.')
    else:
        st.subheader('💰 Prediction')
        st.write(f'The predicted selling price of the car is: **{prediction[0]:.2f} lakh**')

    if st.button('End Prediction'):
        st.session_state.page = 'end'

def end_page():
    st.title("🎉 Thank You!")
    st.markdown("""
    Hope you enjoyed using the Car Selling Price Prediction App.  
    If you have any feedback, feel free to share!
    """)
    if st.button('Restart'):
        st.session_state.page = 'welcome'

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

# Page navigation
if st.session_state.page == 'welcome':
    welcome_page()
elif st.session_state.page == 'main':
    main_page()
elif st.session_state.page == 'end':
    end_page()

# Add additional styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f0f0;
    }
    .sidebar .sidebar-content {
        background: #f5f5f5;
    }
    h1 {
        color: #4a4a4a;
    }
    h2, h3 {
        color: #333333;
    }
</style>
""", unsafe_allow_html=True)

