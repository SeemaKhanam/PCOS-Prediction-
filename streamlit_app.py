import streamlit as st
import pandas as pd
import numpy as np


st.title(' 👩‍🦰 PCOS Prediction')

st.info('This app will predict whether you have PCOS')

with st.expander('Data'):
  st.write('**Raw Data**')
  df=pd.read_csv("https://raw.githubusercontent.com/SeemaKhanam/dataset/refs/heads/main/Cleaned-Data.csv")
  #Selecting only relevent features 
  columns_to_drop = ['Height_ft', 'Vegetrian', 'Diet_Fats', 'Diet_Sweets', 'Diet_Fried_Food', 
                   'Diet_Tea_Coffee', 'Diet_Multivitamin', 'Diet_Bread_Cereals', 'Age', 
                   'Marital_Status', 'Exercise_Frequency', 'Exercise_Type', 'Exercise_Duration', 
                   'Smoking', 'Childhood_Trauma', 'Cardiovascular_Disease','Conception_Difficulty','Diet_Bread_Cereals','Diet_Milk_Products','Diet_Fruits	Diet_Vegetables','Diet_Starchy_Vegetables','Diet_NonStarchy_Vegetables','Diet_Fats','Diet_Sweets','Diet_Fried_Food','Diet_Tea_Coffee','Diet_Multivitamin','Vegetarian',
                    'Diet_Fruits','Diet_Vegetables','Sleep_Hours']


  
  df = df.drop(columns_to_drop, axis=1, errors='ignore')
  df
 
  #Splitting x and y 
  X=df.drop(['PCOS'],axis=1)
  y=df['PCOS']
  st.write("**X**")
  X
  st.write("**Y**")
  y
with st.sidebar:
  st.header("Input Features")
  Weight_kg=st.number_input("**Weight (Kg)**")
  st.write("The current number is ",Weight_kg)

  Family_History_PCOS=st.selectbox('**Family History PCOS**',('Yes','No'))
  Menstrual_Irregularity=st.selectbox('**Menstrual Irregularty**',('Yes','No'))
  Hormonal_Imbalance=st.selectbox('**Hormonal Imbalance**',('Yes','No'))
  Hyperandrogenism=st.selectbox('**Hyperandrogenism**',('Yes','No'))
  Hirsutism=st.selectbox('**Hirsutism**',('Yes','No'))
  Mental_Health=st.selectbox('**Mental Health**',('Yes','No'))
  Insulin_Resistance=st.selectbox('**Insulin Resistance**',('Yes','No'))
  Diabetes=st.selectbox("**Diabetes**",('Yes','No'))
  Stress_Level=st.selectbox('**Stress_Level**',('Yes','No'))
  Exercise_Benefit=st.selectbox('**Excersise Benefit**',('Somewhat','Not at All','Not Much'))
  PCOS_Medication=st.text_input("")
  st.write(PCOS_Medication)




  


  
