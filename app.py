import streamlit as st
from fastai.vision.all import *
import pathlib

# Handle pathlib compatibility for different OS
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Title of the app
st.title("Transport klasifikatsiya qiluvchi model")

# File uploader
file = st.file_uploader("Rasm yuklash", type=["png", "jpeg", "jpg", "svg"])

if file is not None:
    try:
        # Convert the uploaded file to a PIL image
        img = PILImage.create(file)
        
        # Display the uploaded image
        st.image(img, caption="Yuklangan rasm", use_column_width=True)
        
        # Load the model
        try:
            model = load_learner("transport_model.pkl")
            
            # Make prediction
            pred, pred_idx, probs = model.predict(img)
            prob_percentage = probs[pred_idx] * 100
            st.success(f"Bashorat: {pred}")
            st.info(f"Ehtimollik: {prob_percentage:.1f}%")
        except Exception as e:
            st.error(f"Modelni yuklashda yoki bashorat qilishda xato: {e}")
    except Exception as e:
        st.error(f"Rasmni yuklashda xato: {e}")
else:
    st.info("Iltimos, rasm yuklang")
