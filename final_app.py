import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Income Prediction", layout="wide")

@st.cache_resource
def load_models():
    models = {}
    try:
        models['Decision Tree'] = joblib.load('decision_tree_8features.pkl')
        models['Random Forest'] = joblib.load('random_forest_8features.pkl')
        models['Logistic Regression'] = joblib.load('logistic_regression_8features.pkl')
        models['KNN'] = joblib.load('knn_8features.pkl')
        models['SVM'] = joblib.load('svm_8features.pkl')
        st.success("5 models loaded successfully!")
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return models

models = load_models()

# =========================
# Metrics
# =========================
f1_scores = {
    'Decision Tree': 0.6467,
    'Random Forest': 0.6858,
    'Logistic Regression': 0.5510,
    'KNN': 0.6762,
    'SVM': 0.6654
}

accuracies = {
    'Decision Tree': 0.8551,
    'Random Forest': 0.8653,
    'Logistic Regression': 0.8228,
    'KNN': 0.8503,
    'SVM': 0.8623
}

recalls = {
    'Decision Tree': 0.582,
    'Random Forest': 0.697,
    'Logistic Regression': 0.708,
    'KNN': 0.721,
    'SVM': 0.806
}

# =========================
# Encoding Maps
# =========================
education_map = {'Bachelors':0, 'HS-grad':1, 'Masters':2, 'Doctorate':3, 'Some-college':4}
marital_map = {'Never-married':0, 'Married-civ-spouse':1, 'Divorced':2, 'Separated':3, 'Widowed':4}
occupation_map = {'Tech-support':0, 'Craft-repair':1, 'Other-service':2, 'Sales':3, 'Exec-managerial':4}
sex_map = {'Male':1, 'Female':0}

# =========================
# Navigation
# =========================
if "page" not in st.session_state:
    st.session_state.page = "home"

# =========================
# Sidebar
# =========================
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

model_list = ["Decision Tree", "Random Forest", "Logistic Regression", "KNN", "SVM", "All Models"]
selected_model = st.sidebar.selectbox("Choose the model:", options=model_list)

# =========================
# Home Page
# =========================
if st.session_state.page == "home":
    st.title("Income Prediction Web App")
    st.write("Welcome! Click below to start the prediction process.")

    if st.button("Go to Prediction"):
        st.session_state.page = "prediction"
        st.rerun()

# =========================
# Prediction Page
# =========================
elif st.session_state.page == "prediction":

    st.title("Model Prediction")
    st.info(f"Currently using: {selected_model}")

    st.subheader("Enter Details:")

    age = st.number_input("Age", min_value=18, max_value=100, value=25)
    hours_per_week = st.number_input("Hours per week", min_value=1, max_value=100, value=40)

    education = st.selectbox("Education", list(education_map.keys()))
    marital_status = st.selectbox("Marital Status", list(marital_map.keys()))
    occupation = st.selectbox("Occupation", list(occupation_map.keys()))
    gender = st.selectbox("Sex", list(sex_map.keys()))

    capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, value=0)

    # =========================
    # Prediction Button
    # =========================
    if st.button("Predict Now"):

        input_data = [[
            age,
            hours_per_week,
            education_map[education],
            marital_map[marital_status],
            occupation_map[occupation],
            sex_map[gender],
            capital_gain,
            capital_loss
        ]]

        # =========================
        # All Models
        # =========================
        if selected_model == "All Models":

            st.subheader("Results from all models:")

            best_model = None
            best_f1 = -1

            for name in model_list[:-1]:

                # فلترة بالـ Recall
                if recalls[name] < 0.6:
                    st.warning(f"{name} ignored (low recall)")
                    continue

                pred = models[name].predict(input_data)[0]
                result = ">50K" if pred == 1 else "<=50K"

                st.write(
                    f"{name} → {result} | "
                    f"F1: {f1_scores[name]:.3f} | "
                    f"Acc: {accuracies[name]:.3f} | "
                    f"Recall: {recalls[name]:.3f}"
                )

                # اختيار الأفضل
                if f1_scores[name] > best_f1:
                    best_f1 = f1_scores[name]
                    best_model = name

            if best_model:
                st.success(f"Best Model: {best_model} ⭐")

        # =========================
        # Single Model
        # =========================
        else:

            pred = models[selected_model].predict(input_data)[0]
            result = ">50K" if pred == 1 else "<=50K"

            st.success(f"Prediction: {result}")
            st.info(
                f"Accuracy: {accuracies[selected_model]:.3f} | "
                f"F1-Score: {f1_scores[selected_model]:.3f}"
            )

    # =========================
    # Back Button
    # =========================
    if st.button("Back to Home"):
        st.session_state.page = "home"
        st.rerun()