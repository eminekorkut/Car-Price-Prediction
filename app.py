# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 09:56:29 2023

@author: HP
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('car_purchasing.sav', 'rb'))

#creating a function for Prediction
def car_purchasing_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    return prediction

def main():
    
    #giving a title
    st.title('Car Purchasing Prediction Web App')
    st.divider()

    st.write("""This app is for getting a price estimation for the customer so a car with the price range given can be advice to the customer""")


    #getting input from the user
    
    age = st.number_input('Age', min_value = 18, max_value=90, value=40, step=1)
    salary = st.number_input('Salary', min_value = 1000, max_value=90000, value=30000, step=1000)
    networth = st.number_input('Net Worth', min_value = 0, max_value=90000, step=2000)


 
    
    #code for prediction
    diagnosis = ''
    
    # getting the input data from the user
    if st.button('Predicted Car Price : '):
        diagnosis = car_purchasing_prediction([age,salary, networth])
        
    st.success(diagnosis)
    

if __name__ == '__main__':
    main()
