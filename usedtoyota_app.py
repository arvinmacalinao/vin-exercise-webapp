import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import joblib

data = pd.read_csv("toyota_regression.csv")

st.title("Used Toyota Cars Pricing Application")
st.write("From the used toyota car data, we built a machine learning model for pricing of used cars.")

#Display 20 rows of the dataframe
df = pd.DataFrame(data)
st.write("list of Used Car Pricing:")
df1 = df.head(20)


#Charts
chart_data = pd.DataFrame(
    data.head(30),
    columns=['price', 'mpg', 'mileage']
)
st.line_chart(chart_data)

chart_data = pd.DataFrame(
    data.head(30),
    columns=['price', 'mpg', 'mileage']
)
st.bar_chart(chart_data)

chart_data = pd.DataFrame(
    data.head(30),
    columns=['price', 'mpg', 'mileage']
)
st.area_chart(chart_data)


# Side bars
st.sidebar.title("Used Toyota Car")
option_sidebar = st.sidebar.selectbox("Would you like to show Data?", ('No', 'Yes'))

if option_sidebar == "Yes":
    st.write(df1)
st.sidebar.write("Please select your car details")


# Select Model
#models = data['model'].unique()
#model_choice = st.sidebar.selectbox('Model:', models)
model_choice = st.sidebar.radio("Model List:" , ('Auris', 'Avensis', 'Aygo', 'Camry', 'C-HR', 'Corolla', 'GT86', 'Hilux', 'IQ', 'Land Cruiser', 'Prius', 'PROACE VERSO', 'RAV4', 'Supra', 'Urban Cruiser', 'Verso', 'Verso-S', 'Yaris'))
if model_choice == 'Auris':
    model_list = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif model_choice == 'Avensis':
     model_list = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif model_choice == 'Aygo':
     model_list = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif model_choice == 'Camry':
     model_list = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif model_choice == 'C-HR':
     model_list = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif model_choice == 'Corolla':
     model_list = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif model_choice == 'GT86':
     model_list = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif model_choice == 'Hilux':
     model_list = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif model_choice == 'IQ':
     model_list = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif model_choice == 'Land Cruiser':
     model_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
elif model_choice == 'Prius':
     model_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
elif model_choice == 'PROACE VERSO':
     model_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
elif model_choice == 'RAV4':
     model_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
elif model_choice == 'Supra':
     model_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
elif model_choice == 'Urban Cruiser':
     model_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
elif model_choice == 'Verso':
     model_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
elif model_choice == 'Verso-S':
     model_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
elif model_choice == 'Yaris':
     model_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        
# Select Year
years = data['year'].unique()
year = st.sidebar.selectbox('Year:', years)

# Select Transmission
#transmission = st.sidebar.radio
transmission = st.sidebar.radio("Transmission:" , ('Manual', 'Automatic', 'Semi-Auto', 'Other'))
if transmission == 'Manual':
    tran_list = [1, 0, 0, 0]
elif transmission =='Automatic':
    tran_list = [0, 1, 0, 0]
elif transmission =='Semi-Auto':
    tran_list = [0, 0, 1, 0]
elif transmission =='Other':
    tran_list = [0, 0, 0, 1]


# Select Mileage
mileage = st.sidebar.slider("Mileage:", 0, 200000, 100000)

fuelType = st.sidebar.radio("FuelType:" , ('Petrol', 'Diesel', 'Hybrid', 'Other'))
if fuelType == 'Petrol':
    fuel_list = [1, 0, 0, 0]
elif fuelType == 'Diesel':
    fuel_list = [0, 1, 0, 0]
elif fuelType == 'Hybrid':
    fuel_list = [0, 0, 1, 0]
elif fuelType == 'Other':
    fuel_list = [0, 0, 0, 1]

mpg = st.sidebar.slider("Miles/Gallon:", 0.1, 100.0, 50.0)

engineSize = st.sidebar.slider("Engine Size:", 0.1, 4.5, 1.5)

st.subheader('Output Car Price')
# Model filename
filename = 'finalized_model.sav'

# Load the model from disk
loaded_model = joblib.load(filename)

# prediction
prediction = round(loaded_model.predict([[year, mileage, mpg, engineSize] + model_list + tran_list + fuel_list])[0])

st.write(f"Your car suggester price is: {prediction}")