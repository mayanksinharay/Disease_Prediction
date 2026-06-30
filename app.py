import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")

model = load_model()

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("❤️ Heart Disease Prediction System")
st.write(
    "Predict the possibility of heart disease using a Machine Learning model."
)

st.markdown("---")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Patient Information")

age = st.sidebar.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=60
)

sex = st.sidebar.selectbox(
    "Sex",
    ["Male", "Female"]
)

cp = st.sidebar.selectbox(
    "Chest Pain Type",
    [0, 1, 2, 3]
)

trestbps = st.sidebar.number_input(
    "Resting Blood Pressure",
    min_value=80,
    max_value=250,
    value=140
)

chol = st.sidebar.number_input(
    "Cholesterol",
    min_value=100,
    max_value=700,
    value=240
)

fbs = st.sidebar.selectbox(
    "Fasting Blood Sugar > 120 mg/dL",
    ["No", "Yes"]
)

restecg = st.sidebar.selectbox(
    "Resting ECG",
    [0, 1, 2]
)

thalach = st.sidebar.number_input(
    "Maximum Heart Rate",
    min_value=50,
    max_value=250,
    value=150
)

exang = st.sidebar.selectbox(
    "Exercise Induced Angina",
    ["No", "Yes"]
)

oldpeak = st.sidebar.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=1.2
)

slope = st.sidebar.selectbox(
    "Slope",
    [0, 1, 2]
)

ca = st.sidebar.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3, 4]
)

thal = st.sidebar.selectbox(
    "Thal",
    [0, 1, 2, 3]
)

# --------------------------------------------------
# Convert Inputs
# --------------------------------------------------

sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

data = pd.DataFrame([{
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

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict", use_container_width=True):

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0]

    confidence = probability[prediction] * 100

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("⚠️ Heart Disease Detected")

    else:

        st.success("✅ No Heart Disease Detected")

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

    if confidence >= 90:
        risk = "Very High Confidence"

    elif confidence >= 75:
        risk = "High Confidence"

    elif confidence >= 60:
        risk = "Moderate Confidence"

    else:
        risk = "Low Confidence"

    st.info(risk)

    st.subheader("Patient Information")

    st.dataframe(data)

    st.warning(
        "This prediction is generated using a Machine Learning model and "
        "should not replace professional medical advice."
    )

st.markdown("---")

st.caption(
    "Developed using Scikit-learn, Pandas and Streamlit."
)