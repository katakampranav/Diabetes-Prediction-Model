import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained model (replace with the best model: SVM)
with open('diabetes_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Custom CSS for styling
st.markdown("""
    <style>
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
    }
    .result {
        font-size: 24px;
        color: #FF5733;
        text-align: center;
    }
    .input-label {
        font-size: 16px;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 5px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .centered-button {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .form-container {
        padding-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<div class="title">Diabetes Risk Prediction</div>', unsafe_allow_html=True)

# Use st.form for grouping inputs and the button together
with st.form(key='diabetes_form', clear_on_submit=False):
    # Input features with improved layout
    st.subheader("Please enter the following details:")
    age = st.number_input('Age', min_value=1, max_value=100, value=25, step=1)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    polyuria = st.selectbox('Polyuria (Frequent Urination)', ['No', 'Yes'])
    polydipsia = st.selectbox('Polydipsia (Excessive Thirst)', ['No', 'Yes'])
    sudden_weight_loss = st.selectbox('Sudden Weight Loss', ['No', 'Yes'])
    weakness = st.selectbox('Weakness', ['No', 'Yes'])
    polyphagia = st.selectbox('Polyphagia (Excessive Hunger)', ['No', 'Yes'])
    genital_thrush = st.selectbox('Genital Thrush', ['No', 'Yes'])
    visual_blurring = st.selectbox('Visual Blurring', ['No', 'Yes'])
    itching = st.selectbox('Itching', ['No', 'Yes'])
    irritability = st.selectbox('Irritability', ['No', 'Yes'])
    delayed_healing = st.selectbox('Delayed Healing', ['No', 'Yes'])
    partial_paresis = st.selectbox('Partial Paresis', ['No', 'Yes'])
    muscle_stiffness = st.selectbox('Muscle Stiffness', ['No', 'Yes'])
    alopecia = st.selectbox('Alopecia', ['No', 'Yes'])
    obesity = st.selectbox('Obesity', ['No', 'Yes'])

    # Encode categorical inputs
    le = LabelEncoder()
    features = [gender, polyuria, polydipsia, sudden_weight_loss, weakness, polyphagia, genital_thrush,
                visual_blurring, itching, irritability, delayed_healing, partial_paresis, muscle_stiffness,
                alopecia, obesity]
    encoded_features = [le.fit_transform([feature])[0] for feature in features]

    # Prepare input features for prediction
    user_input = np.array([[age] + encoded_features])

    # Standardize the input features
    scaler = StandardScaler()
    user_input_scaled = scaler.fit_transform(user_input)

    # Prediction button placed at the bottom inside the form
    predict_button = st.form_submit_button('Predict Diabetes Risk', use_container_width=True)

# Trigger the prediction only after the button is pressed
if predict_button:
    # Prediction
    prediction = model.predict(user_input_scaled)

    # Display the prediction result with custom styling
    if prediction == 1:
        st.markdown('<div class="result">**Result:** The person is at risk of diabetes. Please consult a healthcare professional for further assessment.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result">**Result:** The person is not at risk of diabetes based on the given data.</div>', unsafe_allow_html=True)
