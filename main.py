import streamlit as st
import pandas as pd
import os


external_path = r"C:\Users\3B\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\LocalState\sessions\34BD55A31D9E53AFADB53A18DE1EDC0C65D69A5D\transfers\2026-18\test_data.csv"

def load_data():
    if os.path.exists(external_path):
       return pd.read_csv(external_path)
    return None

df = load_data()


if "page" not in st.session_state:
    st.session_state.page = "home"


st.sidebar.title("Account")

option = st.sidebar.selectbox("Choose", ["Login", "Sign Up"])

if option == "Login":
    st.sidebar.text_input("Username")
    st.sidebar.text_input("Password", type="password")
    st.sidebar.button("Login")
else:
    st.sidebar.text_input("New Username")
    st.sidebar.text_input("New Password", type="password")
    st.sidebar.button("Sign Up")

st.sidebar.markdown("---")
st.sidebar.subheader("Model Selection")


models = ["Decision Tree", "KNN", "Logistic Regression", "RANDOM FOREST", "SVM", "All Models"]
selected_model = st.sidebar.selectbox("Choose the model:", options=models)


if st.session_state.page == "home":
    st.title("Income Prediction Web App")
    st.write("Welcome! Click below to start the prediction process.")

    if st.button("Go to Prediction"):
        st.session_state.page = "prediction"
        st.rerun()


elif st.session_state.page == "prediction":
    st.title("Model Prediction")
    st.info(f"Currently using: *{selected_model}*")
    st.subheader("Enter Details:")
    age = st.number_input("Age", min_value=18, max_value=100, value=25)

    hours_per_week = st.number_input("hours-per-week", min_value=1, max_value=100, value=40)

    education_options = ["Bachelors", "HS-grad", "Masters", "Doctorate", "Some-college"]
    education = st.selectbox("education", education_options)

    marital_options = ["Never-married", "Married-civ-spouse", "Divorced", "Separated", "Widowed"]
    marital_status = st.selectbox("Marital Status", marital_options)

    occupation_options = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial"]
    occupation = st.selectbox("occupation", occupation_options)

    gender = st.selectbox("sex", ["Male", "Female"])

    capital_gain = st.number_input("capital-gain", min_value=0, value=0)

    capital_loss = st.number_input("capital-loss", min_value=0, value=0)

    if st.button("Predict Now"):

        st.success(f"Prediction complete using {selected_model}!")


    if st.button("Back to Home"):
        st.session_state.page = "home"
        st.rerun()

