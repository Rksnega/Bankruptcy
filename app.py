import streamlit as st
import pandas as pd
import pickle
import sklearn

# Load trained Random Forest model
with open("rf_model.pkl", "rb") as file:
    model = pickle.load(file)

# App configuration
st.set_page_config(page_title="Bankruptcy Prediction App", layout="centered")
st.title("💼 Bankruptcy Prediction using Random Forest")

st.write("Select financial indicator values below to predict whether a company is **Bankrupt** or **Non Bankrupt**.")

# Dropdown options
options = [0.0, 0.5, 1.0]

try:
    input_data = {}
    for feature in model.feature_names_in_:
        input_data[feature] = st.selectbox(f"{feature}", options, index=0)

    # Predict button
    if st.button("Predict Bankruptcy"):
        input_df = pd.DataFrame([input_data])
        result = model.predict(input_df)[0]

        st.subheader("Prediction Result:")
        if result == 1:
            st.error("The company is **Bankrupt**")
        else:
            st.success("The company is **Non Bankrupt**")

except Exception as e:

    st.warning("Could not load feature names from the model. Please ensure the model was trained using scikit-learn ≥ 1.0")
