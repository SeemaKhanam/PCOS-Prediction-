import streamlit as st
import pandas as pd
import numpy as np


st.title(' 👩‍🦰 PCOS Prediction')

st.info('This app will predict whether you have PCOS')

with st.expander('Data'):
  st.write('**Raw Data**')
  df=pd.read_csv("https://raw.githubusercontent.com/SeemaKhanam/dataset/refs/heads/main/Cleaned-Data.csv")
  df
