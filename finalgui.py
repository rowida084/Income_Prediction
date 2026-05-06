import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Income Prediction", layout="wide")

# =========================
# Load Models
# =========================
@st.cache_resource
def load_models():
    files = {
        'Decision Tree':       'pipeline_decisiontree.pkl',
        'Random Forest':       'pipeline_randomforest.pkl',
        'Logistic Regression': 'pipeline_logistic.pkl',
        'KNN':                 'pipeline_knn.pkl',
        'SVM':                 'pipeline_svm.pkl',
    }
    loaded = {}
    for name, file in files.items():
        try:
            loaded[name] = joblib.load(file)
        except Exception as e:
            st.error(f"Error loading {name}: {e}")
    return loaded

models = load_models()

# =========================
# Metrics
# =========================
metrics = {
    'Decision Tree':       {'accuracy': 0.8604, 'f1': 0.6578, 'recall': 0.5681},
    'Random Forest':       {'accuracy': 0.8460, 'f1': 0.7065, 'recall': 0.7850},
    'Logistic Regression': {'accuracy': 0.8087, 'f1': 0.6745, 'recall': 0.8391},
    'KNN':                 {'accuracy': 0.8500, 'f1': 0.6650, 'recall': 0.6303},
    'SVM':                 {'accuracy': 0.8372, 'f1': 0.6912, 'recall': 0.7712},
}

# =========================
# Build Input DataFrame
# =========================
def build_input(age, workclass, education_num, marital_status,
                occupation, relationship, gender,
                capital_gain, capital_loss, hours_per_week, native_country):
    return pd.DataFrame([{
        "age":            age,
        "workclass":      workclass,
        "fnlwgt":         0,
        "education":      "",
        "education-num":  education_num,
        "marital-status": marital_status,
        "occupation":     occupation,
        "relationship":   relationship,
        "race":           "",
        "sex":            gender,
        "capital-gain":   capital_gain,
        "capital-loss":   capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country
    }])

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
    st.title("💰 Income Prediction Web App")
    st.write("Welcome! Click below to start the prediction process.")
    if st.button("Go to Prediction"):
        st.session_state.page = "prediction"
        st.rerun()

# =========================
# Prediction Page
# =========================
elif st.session_state.page == "prediction":
    st.title("🔮 Model Prediction")
    st.info(f"Currently using: **{selected_model}**")
    st.subheader("Enter Details:")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=25)
        workclass = st.selectbox("Work Class", [
            "Private", "Local-gov", "Never-worked", "Self-emp-inc",
            "Self-emp-not-inc", "State-gov", "Without-pay"])
        hours_per_week = st.number_input("Hours per week", min_value=1, max_value=100, value=40)
        education_num = st.number_input("Years of Education", min_value=1, max_value=20, value=10)
        marital_status = st.selectbox("Marital Status", [
            "Never-married", "Married-civ-spouse", "Divorced",
            "Separated", "Widowed", "Married-AF-spouse", "Married-spouse-absent"])
        native_country = st.selectbox("Native Country", ["United-States", "Other"])

    with col2:
        occupation = st.selectbox("Occupation", [
            "Armed-Forces", "Craft-repair", "Exec-managerial", "Farming-fishing",
            "Handlers-cleaners", "Machine-op-inspct", "Other-service",
            "Priv-house-serv", "Prof-specialty", "Protective-serv",
            "Sales", "Tech-support", "Transport-moving"])
        gender = st.selectbox("Sex", ["Male", "Female"])
        relationship = st.selectbox("Relationship", [
            "Husband", "Not-in-family", "Other-relative",
            "Own-child", "Unmarried", "Wife"])
        capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
        capital_loss = st.number_input("Capital Loss", min_value=0, value=0)

    st.markdown("---")

    # =========================
    # Predict Button
    # =========================
    if st.button("Predict Now", type="primary"):
        input_df = build_input(age, workclass, education_num, marital_status,
                               occupation, relationship, gender,
                               capital_gain, capital_loss, hours_per_week, native_country)

        # ── All Models ──────────────────────────────────
        if selected_model == "All Models":
            st.subheader("📊 Results from All Models:")

            rows = []

            for name in model_list[:-1]:
                if name not in models:
                    continue

                pred  = models[name].predict(input_df)[0]
                label = ">50K 💰" if pred == 1 else "<=50K"

                rows.append({
                    "No.":        len(rows) + 1,
                    "Model":      name,
                    "Prediction": label,
                    "Accuracy":   metrics[name]['accuracy'],
                    "F1 Score":   metrics[name]['f1'],
                    "Recall":     metrics[name]['recall'],
                    "Note":       ""
                })

            df = pd.DataFrame(rows)
            df = df.sort_values(by="F1 Score", ascending=False).reset_index(drop=True)
            df["No."] = df.index + 1

            # تحديد أفضل موديل
            best_name = df.iloc[0]["Model"]

            # تعديل الـ Note (Recommended + Low Recall)
            df["Note"] = df.apply(
                lambda row:
                    ("⭐ Recommended " if row["Model"] == best_name else "") +
                    ("⚠️ Low Recall" if row["Recall"] < 0.6 else ""),
                axis=1
            )

            st.dataframe(df, use_container_width=True, hide_index=True)

            best_pred = df.iloc[0]["Prediction"]

            st.success(
                f"""
⭐ **Recommended Model: {best_name}**

Chosen based on:
- Highest F1 Score → {metrics[best_name]['f1']:.4f}
- Balanced performance between Precision & Recall

📊 Final Prediction: **{best_pred}**
"""
            )

        # ── Single Model ─────────────────────────────────
        else:
            pred  = models[selected_model].predict(input_df)[0]
            label = ">50K 💰" if pred == 1 else "<=50K"

            if pred == 1:
                st.success(f"💰 Estimated Income: **{label}**")
            else:
                st.warning(f"💰 Estimated Income: **{label}**")

            st.subheader("📈 Model Performance:")
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{metrics[selected_model]['accuracy']:.4f}")
            c2.metric("F1 Score", f"{metrics[selected_model]['f1']:.4f}")
            c3.metric("Recall",   f"{metrics[selected_model]['recall']:.4f}")

    # =========================
    # Back Button
    # =========================
    if st.button("Back to Home"):
        st.session_state.page = "home"
        st.rerun()