import streamlit as st
import pandas as pd
import numpy as np


st.title(' 👩‍🦰 PCOS Prediction')

st.info('This app will predict whether you have PCOS')

with st.expander('Data'):
  st.write('**Raw Data**')
  df=pd.read_csv("https://raw.githubusercontent.com/SeemaKhanam/dataset/refs/heads/main/Cleaned-Data.csv")
  #Selecting only relevent features 
  df=df.drop(['Height_ft','Vegetrian','Diet_Fats','Diet_Sweets','Diet_Fried_Food','Diet_Tea_Coffee','Diet_Multivitamin','Diet_Bread_Cereals','Age','Marital_Status','Exercise_Frequency','Exercise_Type','Exercise_Duration','Smoking','Childhood_Trauma','Cardiovascular_Disease'],axis=1)
  df
  #Splitting x and y 
  X=df.drop(['PCOS'],axis=1)
  y=df['PCOS']
  st.write("**X**")
  X
  st.write("**Y**")
  y
with st.slider:
st.header("Input Features")
Weight_kg	PCOS	Family_History_PCOS	Menstrual_Irregularity	Hormonal_Imbalance	Hyperandrogenism	Hirsutism	Mental_Health	Conception_Difficulty	Insulin_Resistance	...	Diet_Fats	Diet_Sweets	Diet_Fried_Food	Diet_Tea_Coffee	Diet_Multivitamin	Vegetarian	Sleep_Hours	Stress_Level	Exercise_Benefit	PCOS_Medication
  
