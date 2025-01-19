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

# Ensure that variables are initialized and available for the data dictionary
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

# One-hot encode the input DataFrame using the same encoder
input_encoded = OHE.transform(input_df)  # Encode the input features

# Align the input DataFrame to match the training data columns (one-hot encoded features)
expected_columns = OHE.get_feature_names_out(input_df.columns)
input_encoded = pd.DataFrame(input_encoded, columns=expected_columns)

# Ensure all features (Weight, Height, and one-hot encoded features) are in the correct order
input_features = np.hstack([[Weight_kg, Height_ft], input_encoded.values])

# Make the prediction
prediction = LR.predict(input_features.reshape(1, -1))

# Display the prediction result
if prediction[0] == 1:
    st.success("The model predicts that you have PCOS.")
else:
    st.success("The model predicts that you do not have PCOS.")
