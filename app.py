import streamlit as st
import json
import pickle
from sklearn.preprocessing import LabelEncoder
from streamlit_lottie import st_lottie
import numpy as np

# Sidebar with Title
st.sidebar.markdown("<h1 style='text-align: center; font-size: 3em; color: white;'>DIABETES RISK PREDICTION</h1>", unsafe_allow_html=True)

# Function to load Lottie animation from a local file
def load_lottie_file(filepath):
    with open(filepath, "r") as file:
        return json.load(file)

# Load the trained model (replace with the best model: SVM)
with open('diabetes_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load Lottie animation from local file
lottie_animation = load_lottie_file("lottie_img.json")

# Custom CSS to adjust sidebar width, background color, and center Lottie animation
st.markdown(
    """
    <style>
        /* Center the Lottie animation */
        .center {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
        }
    </style>
    """, 
    unsafe_allow_html=True
)

# Center the Lottie animation inside the sidebar
with st.sidebar:
    st.markdown('<div class="center">', unsafe_allow_html=True)
    st_lottie(lottie_animation, speed=1, width=500, height=500, key="lottie")
    st.markdown('</div>', unsafe_allow_html=True)


# Main form
# st.title("Please enter the following details:")

with st.form(key='diabetes_form', clear_on_submit=False):
    # Create two columns for side-by-side input fields
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Age', min_value=1, max_value=100, value=25, step=1)
        gender = st.selectbox('Gender', ['Male', 'Female'])
        polyuria = st.selectbox('Polyuria (Frequent Urination)', ['No', 'Yes'])
        polydipsia = st.selectbox('Polydipsia (Excessive Thirst)', ['No', 'Yes'])
        sudden_weight_loss = st.selectbox('Sudden Weight Loss', ['No', 'Yes'])
        weakness = st.selectbox('Weakness', ['No', 'Yes'])
        polyphagia = st.selectbox('Polyphagia (Excessive Hunger)', ['No', 'Yes'])
        genital_thrush = st.selectbox('Genital Thrush', ['No', 'Yes'])

    with col2:
        visual_blurring = st.selectbox('Visual Blurring', ['No', 'Yes'])
        itching = st.selectbox('Itching', ['No', 'Yes'])
        irritability = st.selectbox('Irritability', ['No', 'Yes'])
        delayed_healing = st.selectbox('Delayed Healing', ['No', 'Yes'])
        partial_paresis = st.selectbox('Partial Paresis', ['No', 'Yes'])
        muscle_stiffness = st.selectbox('Muscle Stiffness', ['No', 'Yes'])
        alopecia = st.selectbox('Alopecia', ['No', 'Yes'])
        obesity = st.selectbox('Obesity', ['No', 'Yes'])

    le = LabelEncoder()
    features = [gender, polyuria, polydipsia, sudden_weight_loss, weakness, polyphagia, genital_thrush,
                visual_blurring, itching, irritability, delayed_healing, partial_paresis, muscle_stiffness,
                alopecia, obesity]
    encoded_features = [le.fit_transform([feature])[0] for feature in features]

    # Prepare input features for prediction
    user_input = np.array([[age] + encoded_features])

    # Centered prediction button
    predict_button = st.form_submit_button('Predict Diabetes Risk', use_container_width=True)

# Trigger the prediction only after the button is pressed
if predict_button:
    prediction = model.predict(user_input)

    # Display the prediction result with custom styling
    if prediction == 1:
        st.markdown('<div class="result">Result: The person is at risk of diabetes. Please consult a healthcare professional for further assessment.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result">**Result: The person is not at risk of diabetes based on the given data.</div>', unsafe_allow_html=True)

# Add custom CSS to center the button
st.markdown(
    """
    <style>
        .stButton > button {
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
    </style>
    """, unsafe_allow_html=True
)
