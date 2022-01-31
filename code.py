import numpy as np
import pandas as pd
import streamlit as st
import pickle

st.title("Medical Insurance charge prediction")

age = st.number_input(label = "Enter Age", value = 20)
# st.write(age)

gender = st.selectbox(label = "Select Gender", options= ["female", "male"])

bmi = 0.0
column1, column2 = st.columns(2)
enter_bmi = column1.checkbox("Enter BMI")
calc_bmi = column2.checkbox("Calculate BMI")

if enter_bmi and calc_bmi:
    st.error("Select one of them")

elif enter_bmi:
    entered_bmi = column1.number_input(label = "Enter BMI", value = 22.222)
    column1.info("Check the other box if you want to calculate your BMI")
    bmi = float(format(entered_bmi, ".3f"))
    column1.write(bmi)

elif calc_bmi:
    height = column2.number_input(label = "Enter Height (in cms)", value = 150.0)
    weight = column2.number_input(label = "Enter weight (in kgs)", value = 50.0)
    column2.info("Check the other box if you want to enter your BMI")
    if height != 0 and weight != 0:

        bmi_result = format((weight / (height*height))*10000, ".3f")
        column2.write(bmi_result)
        bmi = bmi_result

else :
    st.info("Select one of them!")

children = st.number_input(label = "Enter number of children", value = 0)

smoker = st.selectbox(label = "Smoker", options= ["no", "yes"])

region  = st.selectbox(label = "Select Region", options= ["southwest", "southeast", "northwest", "northeast"])

final_feature_list = [int(age), gender, bmi, int(children), smoker, region] 

final_df = pd.DataFrame([final_feature_list], columns = ["age", "sex", "bmi", "children", "smoker" , "region"] )
# st.write(final_df)

file_pipeline = open("pickles/full_pipeline.sav", "rb")
full_pipeline = pickle.load(file_pipeline)

prep_df = full_pipeline.transform(final_df)
file_model = open("pickles/final_model.sav", "rb")
final_model = pickle.load(file_model)

final_result = final_model.predict(prep_df)
st.write('Predicted Medical Charges: ', final_result[0])

df = pd.read_csv('data/insurance.csv')
view_df = st.checkbox('Click to view train Dataset')
if view_df: 
    st.write(df)
