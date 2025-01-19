import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Streamlit Title
st.title(' 👩‍🦰 PCOS Prediction')

st.info('This app will predict whether you have PCOS')

# Data Loading
with st.expander('Data'):
    st.write('**Raw Data**')
    df = pd.read_csv("https://raw.githubusercontent.com/SeemaKhanam/dataset/refs/heads/main/Cleaned-Data.csv")
    
    # Selecting only relevant features 
    columns_to_drop = ['Height_ft', 'Vegetrian', 'Diet_Fats', 'Diet_Sweets', 'Diet_Fried_Food', 
                       'Diet_Tea_Coffee', 'Diet_Multivitamin', 'Diet_Bread_Cereals', 'Age', 
                       'Marital_Status', 'Exercise_Frequency', 'Exercise_Type', 'Exercise_Duration', 
                       'Smoking', 'Childhood_Trauma', 'Cardiovascular_Disease','Conception_Difficulty',
                       'Diet_Bread_Cereals','Diet_Milk_Products','Diet_Fruits','Diet_Vegetables',
                       'Diet_Starchy_Vegetables','Diet_NonStarchy_Vegetables','Diet_Fats','Diet_Sweets',
                       'Diet_Fried_Food','Diet_Tea_Coffee','Diet_Multivitamin','Vegetarian','Sleep_Hours']
    
    df = df.drop(columns_to_drop, axis=1, errors='ignore')
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
    st.write(PCOS_Medication)

# Preprocessing: Label Encoding for target variable
LE = LabelEncoder()
y_new = LE.fit_transform(y)

# One-Hot Encoding for features
OHE = OneHotEncoder(drop='first', sparse_output=False, dtype=np.int32, handle_unknown='ignore')
X_new = X.drop(['Weight_kg'], axis=1)
x_train_new = OHE.fit_transform(X_new)

# Preparing the data for training
weight_array_train = X['Weight_kg'].values.reshape(-1, 1)
t = np.hstack([weight_array_train, x_train_new])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(t, y_new, test_size=0.2, random_state=0)

# Train the Logistic Regression model
LR = LogisticRegression(max_iter=1000)
LR.fit(X_train, y_train)

# Prepare the input for prediction
data = {
    'Family_History_PCOS': Family_History_PCOS,
    'Menstrual_Irregularity': Menstrual_Irregularity,
    'Hormonal_Imbalance': Hormonal_Imbalance,
    'Hyperandrogenism': Hyperandrogenism,
    'Hirsutism': Hirsutism,
    'Mental_Health': Mental_Health,
    'Insulin_Resistance': Insulin_Resistance,
    'Diabetes': Diabetes,
    'Stress_Level': Stress_Level,
    'Exercise_Benefit': Exercise_Benefit
}

input_df = pd.DataFrame(data, index=[0])

# One-Hot Encoding for the input
input_encoded = pd.get_dummies(input_df, drop_first=True)

# Reindex input_encoded to match the training data columns
input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

# Adding the weight feature for prediction
weight_array_input = np.array([Weight_kg]).reshape(-1, 1)
input_encoded = np.hstack([weight_array_input, input_encoded])

# Predict the outcome
y_pred = LR.predict(input_encoded)
prediction_proba = LR.predict_proba(input_encoded)

# Display the prediction result
df_prediction = pd.DataFrame(prediction_proba, columns=['No', 'Yes'])
df_prediction = df_prediction.rename(columns={0: 'No', 1: 'Yes'})

st.subheader("Diagnosis")
st.dataframe(df_prediction)

# Display final prediction
op = np.array(['Yes', 'No'])
st.success(f"Prediction: {op[y_pred][0]}")
