import streamlit as st
import numpy as np
import joblib
import time

# Set page config for background color
st.set_page_config(page_title='Bill Prediction', page_icon=':money_with_wings:', layout='centered', initial_sidebar_state='auto')

# Custom CSS for background color and styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #e0f7fa;
    }
    .stButton>button {
        background-color: #00796b;
        color: white;
        font-weight: bold;
    }
    .stNumberInput>div>input {
        background-color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained model
model = joblib.load('insurance_model.pkl')

st.title('Bill Prediction :money_with_wings:')
st.write('Enter the values for **Source Load (S_LOAD)**, **Consumer Load (C_LOAD)**, and **Total Unit (TO_UNIT)** to predict the **Total Bill (TOTAL_BILL)**.')

s_load = st.number_input('Source Load (S_LOAD)', min_value=0.0, format="%f")
c_load = st.number_input('Consumer Load (C_LOAD)', min_value=0.0, format="%f")
to_unit = st.number_input('Total Unit (TO_UNIT)', min_value=0.0, format="%f")

if st.button('Predict'):
    with st.spinner('Calculating prediction...'):
        time.sleep(1.5)  # Animation effect
        X_new = np.array([[s_load, c_load, to_unit]])
        prediction = model.predict(X_new)
    st.balloons()  # Animation
    st.success(f'Predicted Total Bill (TOTAL_BILL): {prediction[0]:.2f}')
