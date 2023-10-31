import streamlit as st
import pickle

model = pickle.load(open('prediksi_diabetes_2.sav', 'rb'))

st.title('Diabetes Prediction App')

st.write('Please enter the following details for the prediction')
# insert the data in this order numbers of times pregnant, glucose level, bloodPressure, SkinThickness, insulin, BMI, Diabetes Predigree Function, age
col1, col2 = st.columns(2)
with col1:
    glucose = st.number_input('Glucose Level', 0, 200, 1)
    bloodPressure = st.number_input('Blood Pressure', 0, 130, 1)
    insulin = st.number_input('Insulin', 0, 846, 3)
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', 0.08, 2.42, 1.00)
with col2:
    Pregnancies = st.number_input('Number of Pregnancies', 0, 20, 1)
    SkinThickness = st.number_input('Skin Thickness', 0, 100, 1)
    BMI = st.number_input('BMI', 0.0, 68.0, 20.0)
    Age = st.number_input('Age', 10, 90, 21)

if st.button('Predict'):
    X = [[Pregnancies, glucose, bloodPressure, SkinThickness, insulin, BMI, DiabetesPedigreeFunction, Age]]
    y = model.predict(X)
    if y[0] == 0:
        st.write('You are not likely to have diabetes')
    else:
        st.write('You are likely to have diabetes')