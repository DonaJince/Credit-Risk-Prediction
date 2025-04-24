import streamlit as st
import joblib
import pandas as pd

# Load the scaler, model, and lencoders
scaler = joblib.load('scaler.pkl')
model = joblib.load('rf_model.pkl')
lencoders = joblib.load('lencoders.pkl')

# Set the title of the app
st.title("Credit Risk Prediction")

# Input fields for numerical features
age = st.number_input("Age", min_value=18, max_value=100, value=30)
job = st.number_input("Job", min_value=0, max_value=3, value=0)  # Assuming 'Job' is numerical
credit_amount = st.number_input("Credit amount", min_value=0, value=1000)
duration = st.number_input("Duration", min_value=0, value=12)

# Input fields for categorical features with label encoding
sex = st.selectbox("Sex", options=list(lencoders['Sex'].classes_), key='sex')
housing = st.selectbox("Housing", options=list(lencoders['Housing'].classes_), key='housing')
saving_accounts = st.selectbox("Saving accounts", options=list(lencoders['Saving accounts'].classes_), key='saving_accounts')
checking_account = st.selectbox("Checking account", options=list(lencoders['Checking account'].classes_), key='checking_account')
purpose = st.selectbox("Purpose", options=list(lencoders['Purpose'].classes_), key='purpose')

# Create a button to trigger prediction
if st.button("Predict"):
    # Encode categorical values
    sex_encoded = lencoders['Sex'].transform([sex])[0]
    housing_encoded = lencoders['Housing'].transform([housing])[0]
    saving_accounts_encoded = lencoders['Saving accounts'].transform([saving_accounts])[0]
    checking_account_encoded = lencoders['Checking account'].transform([checking_account])[0]
    purpose_encoded = lencoders['Purpose'].transform([purpose])[0]

    # Create a DataFrame from the input values
    input_data = pd.DataFrame([[age, sex_encoded, job, housing_encoded, saving_accounts_encoded, checking_account_encoded, credit_amount, duration, purpose_encoded]],
                             columns=['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Purpose'])
    
    # Scale the input data using the loaded scaler
    scaled_data = scaler.transform(input_data)
    
    # Make the prediction using the loaded model
    prediction = model.predict(scaled_data)[0]
    
    # Display the prediction
    if prediction == 0:
        st.success("Good Risk")
    else:
        st.error("Bad Risk")