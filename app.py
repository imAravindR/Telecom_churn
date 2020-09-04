# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 17:51:04 2020

@author: imarv
"""


# load libraries
import numpy as np
import pandas as pd
import pickle
import PIL
import streamlit as st

#import the data
data = pd.read_csv("Train_churn.csv")

with open("xgbfiles.pickle", "rb") as f:
    sc = pickle.load(f)
    le_state = pickle.load(f)
    le_city = pickle.load(f)
    le_county = pickle.load(f)
    le_plan = pickle.load(f)
    le_event_type = pickle.load(f)
    best_xgb = pickle.load(f)
    


gender = 'M'
state = 'Missouri'
city = 'Brookline'
county = 'Greene'
amount = 1030
plan = "{'25_Mbps'}"
event_type = "['ticket','order']"
daystoresolve = 60
event_date_gap = 410
age = 63
service = 1

gender = 'M'
state = 'Texas'
city = 'Dallas'
county = 'Dallas'
amount = 665
plan = "{'1_Gbps', '300_Mbps', '25_Mbps'}"
event_type = "['ticket','order']"
daystoresolve = 15
event_date_gap = 360
age = 20
service = 1


cols = ['gender','state','city','county','amount','plan','event_type','daystoresolve','event_date_gap','age','service']

def data_preparation(gender,state,city,county,amount,plan,event_type,daystoresolve,event_date_gap,age,service):
    
    # encoding gender
    gender = 1 if (gender.lower() == 'm') else 0
    
    # state
    state = le_state.transform([state])[0]
    
    # city
    city = le_city.transform([city])[0]
    
    # county
    county = le_county.transform([county])[0]
    
    # plan
    plan = le_plan.transform([plan])[0]
    
    # event_type
    # le_event_type.classes_
    if event_type == "['ticket','order']":
        event_type = 2
    elif event_type == "['order']":
        event_type = 1
    else:
        event_type = 0
    # event_type = le_event_type.transform([event_type])    
    
    # Standardization
    test_std = sc.transform(np.array([amount,daystoresolve,event_date_gap,age,service]).reshape(1, -1))[0]
    
    amount,daystoresolve,event_date_gap,age,service = test_std[0],test_std[1],test_std[2],test_std[3],test_std[4]

    
    test = [gender,state,city,county,amount,plan,event_type,daystoresolve,event_date_gap,age,service]
    
    
    return np.array(test).reshape(1, -1)

def predict_churn(gender,state,city,county,amount,plan,event_type,daystoresolve,event_date_gap,age,service):
    
    test = data_preparation(gender,state,city,county,amount,plan,event_type,daystoresolve,event_date_gap,age,service)
    test_df =  pd.DataFrame(test,columns = cols)
    y_pred_proba = best_xgb.predict_proba(test_df)[:,1]
    print(y_pred_proba)
    return y_pred_proba    

def main():
    st.title("Telecom Churn Predictor")
    st.subheader("Created by: Aravind R")
    #checking the data
    st.write("Will the customer stay?")
    check_data = st.checkbox("Data sample")
    if check_data:
        st.write(data.head())
    st.write("Using Machine Learning lets try to predict Churn")
    
    #input the numbers
    gender     = st.radio("Gender", data.gender.unique())
    state = st.selectbox("State", list(data.state.unique()),0)
    city = st.selectbox("City", list(data.city[data.state == state].unique()),0)
    county = st.selectbox("County", list(data.county[data.state == state].unique()),0)
    amount = st.slider("Bill Amount",int(data.amount.min()),int(data.amount.max()),int(data.amount.mean()))
    plan  = st.selectbox("Plan name", list(data.plan.unique()),0)
    event_type= st.radio("Event type",data.event_type.unique())
    daystoresolve  = st.slider("# days to resolve ticket",int(data.daystoresolve.min()),int(data.daystoresolve.max()),int(data.daystoresolve.mean()))
    event_date_gap  = st.slider("# days between events",int(data.event_date_gap.min()),int(data.event_date_gap.max()),int(data.event_date_gap.mean()))
    age     = st.slider("Customer's age?",int(data.age.min()),int(data.age.max()),int(data.age.mean()))
    service = st.selectbox("Service (Years from DoJ)",list(data.service.unique()),0)
    
    
    prediction_proba = predict_churn(gender,state,city,county,amount,plan,event_type,daystoresolve,event_date_gap,age,service)
    prediction = 'Yes' if prediction_proba > 0.45 else 'No'
    #checking prediction house price
    if st.button("Predict!"):
        st.header("Will the customer leave in the near future: {}".format(prediction))
        st.subheader("Customer probability to leave: {}".format(str(int(prediction_proba[0]*100))+'%'))

if __name__=='__main__':
    main()    
    
    
