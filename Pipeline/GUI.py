import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score, f1_score, recall_score

st.set_page_config(page_title="Income Prediction", layout="wide")


# =========================
# Load Models
# =========================
@st.cache_resource
def load_models():
    files = {
        "Decision Tree": "pipeline_decisiontree.pkl",
        "Random Forest": "pipeline_randomforest.pkl",
        "Logistic Regression": "pipeline_logistic.pkl",
        "KNN": "pipeline_knn.pkl",
        "SVM": "pipeline_svm.pkl",
    }
    loaded = {}
    for name, file in files.items():
        try:
            loaded[name] = joblib.load(file)
        except Exception as e:
            st.error(f"Error loading {name}: {e}")
    return loaded


# =========================
# Compute Metrics from Test Data
# =========================
@st.cache_data
def compute_metrics(_models):
    try:
        test = pd.read_csv("test_data.csv")
        test.rename(columns={"Income ": "Income"}, inplace=True)
        test["Income"] = (
            test["Income"].astype(str).str.strip().str.replace(".", "", regex=False)
        )
        test["Income"] = test["Income"].map({">50K": 1, "<=50K": 0})

        X_test = test.drop("Income", axis=1)
        y_test = test["Income"]

        metrics = {}
        for name, pipeline in _models.items():
            y_pred = pipeline.predict(X_test)
            metrics[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
            }
        return metrics

    except Exception as e:
        st.error(f"Error computing metrics: {e}")
        return {}


# =========================
# Build Input DataFrame
# =========================
def build_input(
    age,
    workclass,
    education_num,
    marital_status,
    occupation,
    relationship,
    gender,
    capital_gain,
    capital_loss,
    hours_per_week,
    native_country,
):
    return pd.DataFrame(
        [
            {
                "age": age,
                "workclass": workclass,
                "fnlwgt": 0,
                "education": "",
                "education-num": education_num,
                "marital-status": marital_status,
                "occupation": occupation,
                "relationship": relationship,
                "race": "",
                "sex": gender,
                "capital-gain": capital_gain,
                "capital-loss": capital_loss,
                "hours-per-week": hours_per_week,
                "native-country": native_country,
            }
        ]
    )


# ==============
# get confidence
# =============
def get_confidence(model, input_df):
    probs = model.predict_proba(input_df)[0]

    prob_low = probs[0]  # <=50K
    prob_high = probs[1]  # >50K

    # نحدد القرار
    if prob_high >= 0.5:
        label = ">50K 💰"
        confidence = prob_high
    else:
        label = "<=50K 💸"
        confidence = prob_low

    return label, confidence


# =========================
# Load Models & Metrics&dataframe for metrics
# =========================
models = load_models()
metrics = compute_metrics(models)
metrics_df = pd.DataFrame(metrics).T

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
model_list = [
    "Decision Tree",
    "Random Forest",
    "Logistic Regression",
    "KNN",
    "SVM",
    "All Models",
]
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
        workclass = st.selectbox(
            "Work Class",
           ["Private", "Self-empnot-inc","Self-emp-inc",
                      "Federalgov", "Local-gov", "State-gov",
                      "Without-pay", "Never-worked"],
        )
        hours_per_week = st.number_input(
            "Hours per week", min_value=1, max_value=100, value=40
        )
        education_num = st.number_input(
            "Years of Education", min_value=1, max_value=30, value=10
        )
        marital_status = st.selectbox(
            "Marital Status",
            [
                "Never-married",
                "Married-civ-spouse",
                "Divorced",
                "Separated",
                "Widowed",
                "Married-AF-spouse",
                "Married-spouse-absent",
            ],
        )
        native_country = st.selectbox("Native Country", ["United-States", "Cambodia", "England",
                           "Puerto-Rico", "Canada", "Germany", "India",
                           "Japan", "Greece", "South", "China",
                           "Cuba", "Iran", "Honduras",
                           "Philippines", "Italy", "Poland",
                           "Jamaica", "Vietnam", "Mexico",
                           "Portugal", "Ireland","France",
                           "Dominican-Republic"," Laos",
                           "Ecuador","Taiwan", "Haiti",
                           "Columbia", "Hungary",
                           "Guatemala", "Nicaragua",
                           "Scotland", "Thailand", "Yugoslavia",
                           "El-Salvador",
                           "Trinadad&Tobago", "Peru", "Hong",
                           "Holand-Netherlands","other"],)

    with col2:
        occupation = st.selectbox(
            "Occupation",
            ["Admclerical","Armed-Forces","Craft-repair","Exec-managerial","Farming-fishing","Handlers-cleaners",
                          "Machine-op-inspct","Priv-house-serv","Prof-specialty","Protective-serv",
                          "Sales","Tech-support","Transport-moving","Other-service"],
        )
        gender = st.selectbox("Sex", ["Male", "Female"])
        relationship = st.selectbox(
            "Relationship",
            [
                "Husband",
                "Not-in-family",
                "Other-relative",
                "Own-child",
                "Unmarried",
                "Wife",
            ],
        )
        capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
        capital_loss = st.number_input("Capital Loss", min_value=0, value=0)

        education = st.selectbox(  
            "Education" ,
              ["Bachelors", "Somecollege", "11th", "HS-grad",
                         "Profschool", "Assoc-acdm", "Assoc-voc",
                         "9th", "7th-8th", "12th", "Masters","1st-4th",
                         "10th", "Doctorate", "5th-6th", "Preschool"],
                         )

    st.markdown("---")

    # =========================
    # Predict Button
    # =========================
    if st.button("Predict Now", type="primary"):
        input_df = build_input(
            age,
            workclass,
            education_num,
            marital_status,
            occupation,
            relationship,
            gender,
            capital_gain,
            capital_loss,
            hours_per_week,
            native_country,
        )

        # ── All Models ──────────────────────────────────
        if selected_model == "All Models":
            st.subheader("📊 Results from All Models:")

            rows = []

            for name in model_list[:-1]:
                if name not in models:
                    continue

                label, confidence = get_confidence(models[name], input_df)
                confidence_percent = f"{confidence:.2%}"

                # ================== Add confidence =====================

                rows.append(
                    {
                        "No.": len(rows) + 1,
                        "Model": name,
                        "Prediction": label,
                        "Confidence": confidence_percent,
                        "Accuracy": metrics[name]["accuracy"],
                        "F1 Score": metrics[name]["f1"],
                        "Recall": metrics[name]["recall"],
                        "Note": "",
                    }
                )

            df = pd.DataFrame(rows)
            df = df.sort_values(by="F1 Score", ascending=False).reset_index(drop=True)
            df["No."] = df.index + 1

            best_name = df.iloc[0]["Model"]
            best_pred = df.iloc[0]["Prediction"]

            df["Note"] = df.apply(
                lambda row: ("⭐ Recommended " if row["Model"] == best_name else "")
                + ("⚠️ Low Recall" if row["Recall"] < 0.6 else ""),
                axis=1,
            )

            st.dataframe(df, use_container_width=True, hide_index=True)
            # ===================== Model performance Comparsion=================

            st.subheader("📊 Models Performance Comparison")

            fig, ax = plt.subplots(figsize=(10, 5))

            metrics_df[["accuracy", "f1", "recall"]].plot(kind="bar", ax=ax)

            ax.set_ylabel("Score")
            ax.set_xlabel("Models")
            ax.set_title("Comparison Between Models")

            st.pyplot(fig)

            st.success(f"""
⭐ **Recommended Model: {best_name}**

Chosen based on:
- Highest F1 Score → {metrics[best_name]['f1']:.4f}
- Balanced performance between Precision & Recall

📊 Final Prediction: **{best_pred}**
""")

        # ── Single Model ─────────────────────────────────

        else:

            label, confidence = get_confidence(models[selected_model], input_df)

            if label == ">50K 💰":
                st.success(f"💰 Estimated Income: **{label}**")
            else:
                st.warning(f"💸 Estimated Income: **{label}**")

            st.subheader("📈 Model Performance:")

            c1, c2, c3 = st.columns(3)

            c1.metric("Accuracy", f"{metrics[selected_model]['accuracy']:.4f}")

            c2.metric("F1 Score", f"{metrics[selected_model]['f1']:.4f}")

            c3.metric("Recall", f"{metrics[selected_model]['recall']:.4f}")

            st.info(f"🔍 Model Confidence: {confidence:.2%}")

            st.progress(float(confidence))
            values = metrics_df.loc[selected_model,["accuracy" , "f1" , "recall"] ]
            fig , ax = plt.subplots(figsize=(6,3))
            ax.barh(values.index ,values.values ,color=["skyblue","lightgreen", "salmon"] ,height=0.4)
            ax.set_xlim(0,1)
            ax.set_title(f"{selected_model} performance")
            ax.set_xlabel("Score")
            for i , v in enumerate(values.values):
              ax.text(v+0.01,i ,f"{v:0.2f}" , va= "center")            
            st.pyplot(fig)
    # =========================
    # Back Button
    # =========================
    if st.button("Back to Home"):
        st.session_state.page = "home"
        st.rerun()
