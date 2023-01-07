import streamlit as st
import pickle
import pandas as pd
import joblib

st.header("CAR PRICE PREDICTION")

import pandas as pd
df2=pd.read_csv("autoscout24-germany-dataset.csv")
df2.head()
st.table(df2.head())

from PIL import Image
img=Image.open("araba23-696x464.jpg")
st.image(img, width=350)

st.sidebar.title("Please select the features of the car.")

mileage = st.sidebar.slider("What is the mileage of your car?", 0, 300000, step=5000)

fuel=st.sidebar.radio("What is the fuel type of your car", ['Diesel', 'Gasolinel', 'Electric'])

gear=st.sidebar.selectbox("What is the gearing type of your car", ['Automatic', 'Manual', 'Semi-automatic'])

offerType=st.sidebar.selectbox("What is the offer type of your car", ['Used', 'Manual', 'Niew'])

hp = st.sidebar.slider("What is the hp of your car?", 1, 850, step=5)

year = st.sidebar.slider("What is the age of your car?", 0, 30, step=1)

make = st.sidebar.selectbox("What is the make of your car?",['BMW', 'Volkswagen', 'Renault', 'Peugeot', 'Toyota', 'Opel',
       'Mazda', 'Ford', 'Chevrolet', 'Audi', 'Kia', 'Dacia',
       'Mercedes-Benz', 'MINI', 'Hyundai', 'SEAT', 'Skoda', 'Citroen',
       'Suzuki', 'SsangYong', 'smart', 'Fiat', 'Nissan', 'Honda',
       'Mitsubishi', 'Volvo', 'Land', 'Alfa', 'Jeep', 'Subaru', 'Abarth',
       'Lada', 'Cupra'])

my_dict = {
    
    "mileage":  mileage, 
    "fuel":  fuel, 
    "gear":  gear, 
    "offerType":  offerType,  
    "hp":  hp,
    "year": year,
    "make":  make

}


df = pd.DataFrame.from_dict([my_dict])

st.subheader("Your Car Specs.")
st.dataframe(df)

columns = ['mileage', 'hp', 'year',  'make_Abarth', 'make_Alfa',
       'make_Audi', 'make_BMW', 'make_Chevrolet', 'make_Citroen', 'make_Cupra',
       'make_Dacia', 'make_Fiat', 'make_Ford', 'make_Honda', 'make_Hyundai',
       'make_Jeep', 'make_Kia', 'make_Lada', 'make_Land', 'make_MINI',
       'make_Mazda', 'make_Mercedes-Benz', 'make_Mitsubishi', 'make_Nissan',
       'make_Opel', 'make_Peugeot', 'make_Renault', 'make_SEAT', 'make_Skoda',
       'make_SsangYong', 'make_Subaru', 'make_Suzuki', 'make_Toyota',
       'make_Volkswagen', 'make_Volvo', 'make_smart', 'fuel_Diesel',
       'fuel_Electric', 'fuel_Gas', 'fuel_Gasoline', 'gear_Automatic',
       'gear_Manual', 'gear_Semi-automatic', 'offerType_Demonstration',
       "offerType_Employee's car", 'offerType_New', 'offerType_Pre-registered',
       'offerType_Used']

df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)

model = pickle.load(open("final_model", "rb"))

scaler = joblib.load('final_scaler')

df = scaler.transform(df)

prediction = model.predict(df)

st.success("The estimated price of your car is €{}. ".format(int(prediction[0])))

st.balloons()

