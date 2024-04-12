import streamlit as st
import requests

st.set_page_config(
    page_title="Streamlit App",
    page_icon="",
    layout="wide",
)

st.title("Prédiction Match")

st.image('https://www.francebleu.fr/s3/cruiser-production/2019/03/a98102ae-5e60-4558-b404-f43cfab46b86/1200x680_coupe-du-monde.jpg')

with open('models/model.pkl', 'rb') as f:
   st.download_button('Download CSV', f,file_name='model.pkl') 

