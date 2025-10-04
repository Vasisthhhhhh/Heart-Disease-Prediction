import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os

# Load the saved model + accuracy
model, test_accuracy = joblib.load("model_pipeline.pkl")

# Sidebar info
st.sidebar.title("ℹ️ About")
st.sidebar.write("This app predicts the likelihood of **Heart Disease** using ML.")
st.sidebar.metric("📊 Model Accuracy", f"{test_accuracy*100:.2f}%")

# App Title
st.title("Heart Disease Prediction App")
st.write("Fill out the details below to check the risk of heart disease.")

# Layout: 2 columns for inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    sex = st.selectbox("Sex", ["male", "female"])
    cp = st.selectbox("Chest Pain Type (cp)", [
        "typical angina", "atypical angina", "non-anginal pain", "asymptomatic"
    ])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Serum Cholesterol (chol)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", ["true", "false"])

with col2:
    restecg = st.selectbox("Resting ECG Results (restecg)", [
        "normal", "ST-T abnormality", "left ventricular hypertrophy"
    ])
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=50, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina (exang)", ["yes", "no"])
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment (slope)", ["upsloping", "flat", "downsloping"])
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", ["normal", "fixed defect", "reversible defect"])

# Create dataframe from input
input_data = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}])

# --- Input Validation ---
def validate_inputs(data: pd.DataFrame):
    errors = []

    # Age reasonable range
    if not (20 <= data["age"].iloc[0] <= 100):
        errors.append("Age must be between 20 and 100.")

    # Blood pressure
    if not (80 <= data["trestbps"].iloc[0] <= 200):
        errors.append("Resting blood pressure must be between 80 and 200 mm Hg.")

    # Cholesterol
    if not (100 <= data["chol"].iloc[0] <= 600):
        errors.append("Cholesterol must be between 100 and 600 mg/dl.")

    # Heart rate
    if not (60 <= data["thalach"].iloc[0] <= 220):
        errors.append("Max heart rate must be between 60 and 220 bpm.")

    # ST depression
    if not (0.0 <= data["oldpeak"].iloc[0] <= 6.0):
        errors.append("ST depression (oldpeak) must be between 0.0 and 6.0.")

    return errors

LOG_FILE = "predictions_log.csv"

def log_prediction(input_df, prediction, probability):
    log_row = input_df.copy()
    log_row["prediction"] = prediction
    log_row["probability"] = probability
    log_row["timestamp"] = datetime.utcnow().isoformat()

    # Append to CSV
    if not os.path.exists(LOG_FILE):
        log_row.to_csv(LOG_FILE, index=False)
    else:
        log_row.to_csv(LOG_FILE, mode="a", header=False, index=False)

# Predict button
if st.button("🔍 Predict"):
    errors = validate_inputs(input_data)

    if errors:
        st.error("⚠️ Invalid input(s) detected. Please fix the following:")
        for e in errors:
            st.markdown(f"- **{e}**")
    else:
        # Confirm validation success
        st.success("✅ All inputs look valid!")

        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"⚠️ High Risk: This person likely has heart disease. (Probability: {proba:.2f})")
        else:
            st.success(f"Low Risk: This person is unlikely to have heart disease. (Probability: {proba:.2f})")

            # Log the prediction
            log_prediction(input_data, int(prediction), float(proba))
            st.info("📝 Prediction logged successfully!")

    # Feature importance (only for RandomForest-like models)
    if hasattr(model.named_steps["model"], "feature_importances_"):
        st.subheader("🔎 Feature Importance")
        importances = model.named_steps["model"].feature_importances_
        features = model.named_steps["preprocessor"].get_feature_names_out()
        importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
        importance_df = importance_df.sort_values(by="Importance", ascending=False).head(10)

        st.bar_chart(importance_df.set_index("Feature"))

if os.path.exists(LOG_FILE):
    st.sidebar.subheader("🗂 Recent Predictions")
    logs = pd.read_csv(LOG_FILE).tail(5)
    st.sidebar.dataframe(logs)

# --- Download Full Log ---
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "rb") as f:
        st.sidebar.download_button(
            label="⬇️ Download Full Log (CSV)",
            data=f,
            file_name="predictions_log.csv",
            mime="text/csv"
        )