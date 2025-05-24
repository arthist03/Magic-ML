import streamlit as st
import pandas as pd
import os

#EDA libs Install
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

#ML Model
from pycaret.regression import setup, compare_models, pull, save_model

# Load dataset if already uploaded
if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

# Sidebar with logo, title, navigation, and info
with st.sidebar:
    st.image("https://th.bing.com/th/id/OIP.D9HbBdI_rOzPJJGR54kLbgHaE7?w=249&h=180&c=7&r=0&o=7&cb=iwp2&dpr=1.3&pid=1.7&rm=3")
    st.title("Magic ML 🚀")
    choice = st.radio("Navigation 🧭", ["Upload 📤", "Profiling 📊", "Modelling 🤖", "Download 💾"])
    st.info("This project application helps you build and explore your data. 📈✨")

# Upload tab: upload CSV file and display dataframe
if choice == "Upload 📤":
    st.title("Upload Your Dataset 📥")
    file = st.file_uploader("Upload Your Dataset 🗂️")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=False)  # Save for later use
        st.dataframe(df)


# Profiling tab: show detailed EDA report
if choice == "Profiling 📊":
    st.title("Exploratory Data Analysis 🔍")
    if 'df' in locals():
        profile_df = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
        st_profile_report(profile_df)
    else:
        st.warning("⚠️ Please upload a dataset first.")  # Warn if no data

# Modelling tab: setup PyCaret, compare models, display results
if choice == "Modelling 🤖":
    if 'df' in locals():
        chosen_target = st.selectbox('Choose the Target Column 🎯', df.columns)
        if st.button('Run Modelling ▶️'):
            setup(df, target=chosen_target, session_id=123)
            setup_df = pull()  # Get setup info
            st.dataframe(setup_df)
            best_model = compare_models()  # Find best model
            compare_df = pull()  # Get comparison results
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')  # Save best model
    else:
        st.warning("⚠️ Please upload a dataset first.")

# Download tab: download the saved model file
if choice == "Download 💾":
    if os.path.exists('best_model.pkl'):
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model 📥', f, file_name="best_model.pkl")
    else:
        st.warning("⚠️ No saved model found. Train a model first.")
