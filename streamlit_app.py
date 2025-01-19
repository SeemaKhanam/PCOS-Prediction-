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
 
  PCOS_Medication=st.text_input("**Taking any PCOS medication**","")
  st.write(PCOS_Medication)

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

LE=LabelEncoder()
y_new=LE.fit_transform(y)

OHE=OneHotEncoder(drop='first',sparse_output=False,dtype=np.int32,handle_unknown='ignore')
X_new=X.drop(['Weight_kg'],axis=1)
x_train_new=OHE.fit_transform(X_new)
weight_array = X_train['Weight_kg'].values.reshape(-1, 1)  # Ensure it's 2D
t=np.hstack([weight_array,x_train_new])
data={'Family_History_PCOS':Family_History_PCOS,
  'Menstrual_Irregularity':Menstrual_Irregularity,
  'Hormonal_Imbalance':Hormonal_Imbalance,
  'Hyperandrogenism':Hyperandrogenism,
  'Hirsutism':Hirsutism,
  'Mental_Health':Mental_Health,
  'Insulin_Resistance':Insulin_Resistance,
  'Diabetes':Diabetes,
  'Stress_Level':Stress_Level,
  'Exercise_Benefit':Exercise_Benefit}
input_df=pd.DataFrame(data,index=[0])
encode=df.drop(['Weight_kg'],axis=1)
input=pd.get_dummies(input_df,prefix=encode)
from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(t,y_new)
y_pred=LR.predict(input)
prediction_proba=LR.predict_proba(input)
df_prediction.columns=['Yes','No']
df_prediction,rename(columns={0:'No',1:'Yes'})

st.subheader("Diagnosis")
st.dataframe(df_prediction,
             column_config={
               'Yes':st.column_config.ProgressColumn('Yes',format='%f',width='medium',min_value=0,max_value=1),
               'No':st.column_config.ProgressColumn('No',format='%f',width='medium',min_value=0,max_value=1)})

op=np.array(['Yes','No'])
st.success(str(op[y_pred][0]))


  
