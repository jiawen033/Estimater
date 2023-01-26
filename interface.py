import time
from sklearn import preprocessing
import numpy as np
import pickle
import pandas as pd
from joblib import dump, load

#from flasgger import Swagger
import streamlit as st


def welcome():
    return "Welcome All"


# @app.route('/predict',methods=["Get"])
def predictGen(m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, c1):
    gender_model_clone = load('rf_model.pkl')
    input_data = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15,c1]
    scaled_data = preprocessing.scale(input_data)
    inputtouse = [scaled_data]

    predictionGen = gender_model_clone.predict(inputtouse)

    print(predictionGen)
    return predictionGen


def predictEth(m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, c1):
    eth_model_clone = load('lda_model.pkl')

    input_data = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15,c1]
    scaled_data = preprocessing.scale(input_data)
    inputtouse = [scaled_data]

    predictionEth = eth_model_clone.predict(inputtouse)

    print(predictionEth)
    return predictionEth


def main():
    st.title("Sex and Anchester Estimator")
    html_temp = """
    
    <div style="background-color:grayy;padding:10px">
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    m1 = st.text_input("Age:", "")
    m2 = st.text_input("Maximum Cranial Length(MCL):", "")
    m3 = st.text_input("Lateral Cranial Length(LCL<R>):", "")
    m4 = st.text_input("Lateral Cranial Length(LCL<L>):", "")
    m5 = st.text_input("Nasio Occipital Length(NOL):", "")
    m6 = st.text_input("Maximum Cranial Width((MCW):", "")
    m7 = st.text_input("Biasteronic Width(BAW):", "")
    m8 = st.text_input("InterporionWidth(IPW):", "")
    m9 = st.text_input("Prietal Cord(PC):", "")
    m10 = st.text_input("Occipital Cord(OC):", "")
    m11 = st.text_input("Frontal Cord((FC):", "")
    m12 = st.text_input("Cranial Height(CH):", "")
    m13 = st.text_input("Cranial Base Length(CBL):", "")
    m14 = st.text_input("Foramen Magnum Length(FML):", "")
    m15 = st.text_input("Foramen Magnum Width(FMW):", "")

    result = ""
    if st.button("Predict"):
        result = ""
        width = float(m6)
        length = float(m2)
        ci = (width / length) * 100
        result1 = predictGen(float(m1), float(m2), float(m3), float(m4), float(m5),
                            float(m6), float(m7), float(m8), float(m9), float(m10), float(m11),
                            float(m12), float(m13), float(m14), float(m15), ci)
        result2 = predictEth(float(m1), float(m2), float(m3), float(m4), float(m5),
                            float(m6), float(m7), float(m8), float(m9), float(m10), float(m11),
                            float(m12), float(m13), float(m14), float(m15), ci)
        st.success("The Sex is {}".format(result1[0]))
        ethnicity_map = {'C': 'Chinese', 'I': 'Indian', 'M': 'Malay'}
        st.success("The Anchester is {}".format((ethnicity_map.get(result2[0], "Invalid value"))))



if __name__ == '__main__':
    main()