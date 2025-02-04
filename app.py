import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle


model=tf.keras.models.load_model('model.h5')

with open('laber_encoder_gender.pkl','rb') as file:
    Label_encoder_gender=pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open('scaler.pkl','rb') as file:
    sclar=pickle.load(file)
 

st.title('Customer chrun prediction')

geography=st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender=st.selectbox('Gender',Label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit_score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_product=st.slider('No of Product',0,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_acrive_number=st.selectbox('Is active Number',[0,1])

input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[Label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_product],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_acrive_number],
    'EstimatedSalary':[estimated_salary],
    })

geo_encoded=onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

df=input_data.join(geo_encoded_df)

scaling=sclar.transform(df)

prediction=model.predict(scaling)
prediction_prob=prediction[0][0]

st.write('Prediction_Probablity:-',prediction_prob)

if prediction_prob > 0.5:
    st.write('The customer is likely to Churn')
else:
    st.write('The customer is not likely to churn')
