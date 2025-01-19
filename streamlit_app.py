import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

# Title and introduction
st.title(' 👩‍🦰 PCOS Prediction')
st.info('This app will predict whether you have PCOS')

# Data handling
with st.expander('Data'):
    st.write('**Raw Data**')
    df = pd.read_csv("https://raw.githubusercontent.com/SeemaKhanam/dataset/refs/heads/main/Cleaned-Data.csv")

    # Selecting only relevant features
    columns_to_drop = ['Height_ft', 'Vegetrian', 'Diet_Fats', 'Diet_Sweets', 'Diet_Fried_Food', 
                       'Diet_Tea_Coffee', 'Diet_Multivitamin', 'Diet_Bread_Cereals', 'Age', 
                       'Marital_Status', 'Exercise_Frequency', 'Exercise_Type', 'Exercise_Duration', 
                       'Smoking', 'Childhood_Trauma', 'Cardiovascular_Disease','Conception_Difficulty','Diet_Bread_Cereals','Diet_Milk_Products','Diet_Fruits','Diet_Vegetables','Diet_Starchy_Vegetables','Diet_NonStarchy_Vegetables','Diet_Fats','Diet_Sweets','Diet_Fried_Food','Diet_Tea_Coffee','Diet_Multivitamin','Vegetarian',
                       'Diet_Fruits','Diet_Vegetables','Sleep_Hours']
    df = df.drop(columns_to_drop, axis=1, errors='ignore')

    # Splitting X and y
    X = df.drop(['PCOS'], axis=1)
    y = df['PCOS']

    st.write("**X**")
    st.write(X)
    st.write("**Y**")
    st.write(y)

# Input fields for prediction
with st.sidebar:
    st.header("Input Features")
    Weight_kg = st.number_input("**Weight (Kg)**")
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

# Encoding input features for prediction
data = {'Family_History_PCOS': Family_History_PCOS,
        'Menstrual_Irregularity': Menstrual_Irregularity,
        'Hormonal_Imbalance': Hormonal_Imbalance,
        'Hyperandrogenism': Hyperandrogenism,
        'Hirsutism': Hirsutism,
        'Mental_Health': Mental_Health,
        'Insulin_Resistance': Insulin_Resistance,
        'Diabetes': Diabetes,
        'Stress_Level': Stress_Level,
        'Exercise_Benefit': Exercise_Benefit}
input_df = pd.DataFrame(data, index=[0])

# One-Hot Encoding for categorical input data
encode = input_df.columns
input_encoded = pd.get_dummies(input_df, columns=encode, drop_first=True)

# Ensure that the input data has the same columns as the training data
input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

# Preprocessing for 'Weight_kg' feature
weight_array = np.array([Weight_kg]).reshape(-1, 1)  # Ensuring it is 2D for compatibility
input_encoded = np.hstack([weight_array, input_encoded])

# Train the model
LE = LabelEncoder()
y_new = LE.fit_transform(y)

OHE = OneHotEncoder(drop='first', sparse_output=False, dtype=np.int32, handle_unknown='ignore')
X_new = X.drop(['Weight_kg'], axis=1)
x_train_new = OHE.fit_transform(X_new)

weight_array_train = X['Weight_kg'].values.reshape(-1, 1)
t = np.hstack([weight_array_train, x_train_new])

LR = LogisticRegression()
LR.fit(t, y_new)

# Make prediction
y_pred = LR.predict(input_encoded)
prediction_proba = LR.predict_proba(input_encoded)

# Show results
st.subheader("Diagnosis")
df_prediction = pd.DataFrame(prediction_proba, columns=['No', 'Yes'])
df_prediction = df_prediction.rename(columns={0: 'No', 1: 'Yes'})

st.dataframe(df_prediction)

# Display prediction result
st.success(f"Prediction: {'Yes' if y_pred[0] == 1 else 'No'}")
