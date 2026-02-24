import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load model and scaler
model = load_model("model.h5")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Customer Churn Prediction", page_icon="🏦")

st.title("🏦 Customer Churn Prediction")
st.write("Enter customer details to predict the likelihood of exit.")

# Input fields
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 100, 30)
    tenure = st.slider("Tenure (years)", 0, 10, 3)

with col2:
    credit_score = st.slider("Credit Score", 300, 850, 600)
    balance = st.number_input("Balance", value=0.0)
    num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_credit_card = st.selectbox("Has Credit Card", [0, 1])

with col1:
    is_active_member = st.selectbox("Is Active Member", [0, 1])
    estimated_salary = st.number_input("Estimated Salary", value=50000.0)

# Encode inputs
geo_map = {"France": [1, 0, 0], "Germany": [0, 0, 1], "Spain": [0, 1, 0]}
gender_encoded = 1 if gender == "Male" else 0
geo_encoded = geo_map[geography]

# Construct input array (Matches training order: Geography_Encoded, Gender_Encoded, CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary)
# Based on common patterns and the scaler input requirements.
input_data = np.array(geo_encoded + [gender_encoded, credit_score, age, tenure, balance,
                                     num_of_products, has_credit_card,
                                     is_active_member, estimated_salary]).reshape(1, -1)

# Predict
if st.button("Predict"):
    # Scale input
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    prediction_prob = prediction[0][0]
    prediction_class = (prediction_prob > 0.5).astype(int)
    
    st.divider()
    if prediction_class == 1:
        st.error(f"⚠️ This customer is likely to exit. (Probability: {prediction_prob:.2f})")
    else:
        st.success(f"✅ This customer is likely to stay. (Probability: {prediction_prob:.2f})")
