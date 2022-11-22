from pycaret.regression import load_model, predict_model
import streamlit as st
from readline import set_pre_input_hook
from sys import setprofile
import pandas as pd
import plotly as py
import plotly.figure_factory as ff
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline


model = load_model('Coches_gbr')
input_df = pd.read_csv('nuevo.csv')


def predict_precio (model, input_df):
    predictions_df = predict_model(estimator = model, data = input_df)
    predictions = predictions_df['price'][0]
    return  predictions ## output -> vuelvo a df


prediction = predict_precio(model, input_df) 






st.title('Cochecitos ')
""" Index(['make', 'model', 'version', 'fuel', 'year', 'kms', 'shift',
       'is_professional', 'dealer', 'province', 'publish_date', 'insert_date'],
      dtype='object')"""


model =  st.sidebar.selectbox("Choose car brand",  ('alfa_romeo', 'audi', 'bmw', 'citroen', 'dacia', 'daewoo', 'fiat', 'ford' ,'honda', 'hyundai', 'jeep', 'kia', 'lada', 'lancia', 'land_rover',
         'lexus', 'mazda', 'mercedes', 'mini', 'mitsubishi', 'nissan', 'opel',
         'peugeot', 'renault', 'rover' ,'seat', 'skoda', 'smart', 'subaru', 'suzuki', 'toyota', 'volkswagen', 'volvo'))
fuel = st.sidebar.selectbox("Choose Fuel", ["Gasolina","Diésel", "GLP", "Gasolina y Gas"])
shift = st.sidebar.selectbox("Choose Gear Box",  ['Manual', 'Automatico', 'Libre'])
is_professional = st.sidebar.checkbox("Professional seller") 
year = int(st.sidebar.text_input('Year of registration', '2020'))
province = st.sidebar.selectbox("Choose province",  [ 'alava', 'albacete', 'alicante', 'almeria', 'asturias', 'avila', 'badajoz', 'barcelona', 'burgos', 'caceres', 'cadiz', 'cantabria', 'castellon',
    'ceuta', 'ciudad_real', 'cordoba', 'cuenca', 'girona', 'laspalmas', 'granada', 'guadalajara', 'guipuzcoa', 'huelva', 'huesca', 'illes_balears', 
    'jaen', 'la_rioja', 'leon', 'lleida', 'lugo', 'madrid', 'malaga', 'melilla', 'murcia', 'navarra', 'ourense', 'palencia', 'pontevedra', 'salamanca', 'segovia',
    'sevilla', 'soria', 'tarragona', 'tenerife', 'teruel', 'toledo', 'valencia', 'valladolid', 'vizcaya', 'zamora', 'zaragoza'])
kms = st.sidebar.slider('Select the kilometers',1000,275000,1) 



features = {'model':model,'fuel' : fuel, 'shift' : shift,'is_professional' :is_professional, 'year': year, 'kms': kms, 'province' : province}
df = pd.DataFrame(features, index = [0])
prediction = predict_precio(model, df)       
features_df  = pd.DataFrame([features])




st.table(features_df)

st.header("Predicted Price")
st.title(round(float(prediction )/1000, 2))
st.write(df)

st.button("Realizar Estimación")

if st.button('Predict'):    
        
        st.header('The predicted price is: {}€'.format(prediction))
    
