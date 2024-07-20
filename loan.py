import streamlit as st
import pickle as pk
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# Loading the saved model
model = pk.load(open('Loan_Approval.sav', 'rb'))

# Custom CSS
st.markdown("""
    <style>
    body {
        font-family: Arial, sans-serif;
    }
    h1 {
        color: #4CAF50;
        text-align: center;
        text-decoration: underline;
    }
    h3 {
        text-align: center;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        transition-duration: 0.4s;
    }
    .stButton button:hover {
        background-color: white; 
        color: #4CAF50;
        border: 2px solid #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Loan Approval Prediction Page
st.markdown("<h1>Loan Approval Prediction using ML</h1>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    Married = st.selectbox('Married', ['No', 'Yes'])
    Dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
    Education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    Self_Employed = st.selectbox('Self Employed', ['No', 'Yes'])
    Loan_Amount_Term = st.text_input('Loan Amount Term')
with col2:
    Credit_History = st.text_input('Credit History')
    Property_Area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])
    LoanAmount_log = st.text_input('Log of Loan Amount')
    TotalIncome = st.text_input('Total Income')
    TotalIncome_log = st.text_input('Log of Total Income')

# Real-time validation and feedback
try:
    input_data = [
        float(Gender == 'Male'),
        float(Married == 'Yes'),
        float(Dependents.replace('3+', '3')),
        float(Education == 'Graduate'),
        float(Self_Employed == 'Yes'),
        float(Loan_Amount_Term) if Loan_Amount_Term else 0,
        float(Credit_History) if Credit_History else 0,
        float(Property_Area == 'Urban') + 2 * float(Property_Area == 'Semiurban'),
        float(LoanAmount_log) if LoanAmount_log else 0,
        float(TotalIncome) if TotalIncome else 0,
        float(TotalIncome_log) if TotalIncome_log else 0
    ]
    reshaped_input = np.array(input_data).reshape(1, -1)
    gen_prediction = model.predict(reshaped_input)
    if gen_prediction[0] == 0:
        Predict_diagnosis = 'The person can get a Loan'
        result_color = "green"
    else:
        Predict_diagnosis = 'The person does not get a Loan'
        result_color = "red"
except ValueError as e:
    Predict_diagnosis = f"Invalid input: {e}"
    result_color = "red"

if st.button('Loan Approval Result'):
    st.markdown(f"<h3 style='color: {result_color};'>{Predict_diagnosis}</h3>", unsafe_allow_html=True)
