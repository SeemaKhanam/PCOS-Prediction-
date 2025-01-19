import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Streamlit Title
st.title(' 👩‍🦰 PCOS Prediction')

st.info('This app will predict whether you have PCOS')

# Data Loading
with st.expander('Data'):
    st.write('**Raw Data**')
    df = pd.read_csv("https://raw.githubusercontent.com/SeemaKhanam/dataset/refs/heads/main/Cleaned-Data.csv")
    
    # Selecting only relevant features 
    columns_to_drop = ['Vegetarian', 'Diet_Fats', 'Diet_Sweets', 'Diet_Fried_Food', 
                       'Diet_Tea_Coffee', 'Diet_Multivitamin', 'Diet_Bread_Cereals', 'Age', 
                       'Marital_Status', 'Exercise_Frequency', 'Exercise_Type', 'Exercise_Duration', 
                       'Smoking', 'Childhood_Trauma', 'Cardiovascular_Disease', 'Conception_Difficulty',
                       'Diet_Bread_Cereals', 'Diet_Milk_Products', 'Diet_Fruits', 'Diet_Vegetables',
                       'Diet_Starchy_Vegetables', 'Diet_NonStarchy_Vegetables', 'Sleep_Hours']
    
    df = df.drop(columns=columns_to_drop, axis=1, errors='ignore')
    st.write(df)

    # Splitting X and y 
    X = df.drop(['PCOS'], axis=1)
    y = df['PCOS']
    st.write("**X**")
    st.write(X)
    st.write("**Y**")
    st.write(y)

# Input features for prediction
with st.sidebar:
    st.header("Input Features")
    Weight_kg = st.number_input("**Weight (Kg)**", value=50.0)  # Initialize with a default value
    Height_ft = st.number_input("**Height (ft)**", value=5.5)  # Initialize with a default value
    st.write("The current weight is ", Weight_kg)
    st.write("The current height is ", Height_ft)

    Family_History_PCOS = st.selectbox('**Family History PCOS**', ('Yes', 'No'))
    Menstrual_Irregularity = st.selectbox('**Menstrual Irregularity**', ('Yes', 'No'))
    Hormonal_Imbalance = st.selectbox('**Hormonal Imbalance**', ('Yes', 'No'))
    Hyperandrogenism = st.selectbox('**Hyperandrogenism**', ('Yes', 'No'))
    Hirsutism = st.selectbox('**Hirsutism**', ('Yes', 'No'))
    Mental_Health = st.selectbox('**Mental Health**', ('Yes', 'No'))
    Insulin_Resistance = st.selectbox('**Insulin Resistance**', ('Yes', 'No'))
    Diabetes = st.selectbox("**Diabetes**", ('Yes', 'No'))
    Stress_Level = st.selectbox('**Stress Level**', ('Yes', 'No'))
    Exercise_Benefit = st.selectbox('**Exercise Benefit**', ('Somewhat', 'Not at All', 'Not Much'))
    PCOS_Medication = st.text_input("**Taking any PCOS medication**", "")
    st.write(PCOS_Medication)

# Preprocessing: Label Encoding for target variable
LE = LabelEncoder()
y_new = LE.fit_transform(y)

# One-Hot Encoding for features
OHE = OneHotEncoder(drop='first', sparse_output=False, dtype=np.int32, handle_unknown='ignore')
X_new = X.copy()  # Make a copy to avoid dropping the original data
x_train_new = OHE.fit_transform(X_new)

# Preparing the data for training
weight_array_train = X['Weight_kg'].values.reshape(-1, 1)  # Ensure it's 2D
height_array_train = X['Height_ft'].values.reshape(-1, 1)  # Ensure Height is also 2D
t = np.hstack([weight_array_train, height_array_train, x_train_new])  # Combine Weight, Height, and other features

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(t, y_new, test_size=0.2, random_state=0)

# Train the Logistic Regression model
LR = LogisticRegression(max_iter=1000)
LR.fit(X_train, y_train)

# Prepare the input for prediction
data = {
    'Weight_kg': Weight_kg,
    'Height_ft': Height_ft,
    'Family_History_PCOS': Family_History_PCOS,
    'Menstrual_Irregularity': Menstrual_Irregularity,
    'Hormonal_Imbalance': Hormonal_Imbalance,
    'Hyperandrogenism': Hyperandrogenism,
    'Hirsutism': Hirsutism,
    'Mental_Health': Mental_Health,
    'Insulin_Resistance': Insulin_Resistance,
    'Diabetes': Diabetes,
    'Stress_Level': Stress_Level,
    'Exercise_Benefit': Exercise_Benefit,
    'PCOS_Medication': PCOS_Medication
}

input_df = pd.DataFrame(data, index=[0])  # Create a DataFrame for the input

# Ensure the input DataFrame has the same columns as the training data
input_encoded = OHE.transform(input_df)  # Encode the input features

# Convert input features to 2D array for concatenation
input_features = np.hstack([np.array([[Weight_kg, Height_ft]]), input_encoded])  # Ensure input is 2D

# Make the prediction
prediction = LR.predict(input_features)

# Display the prediction result
if prediction[0] == 1:
    st.success("The model predicts that you have PCOS.")
else:
    st.success("The model predicts that you do not have PCOS.")
