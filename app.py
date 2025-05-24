import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from io import BytesIO

# EDA libs
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# ML libs
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Load dataset if already uploaded
if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

# Sidebar with logo, title, navigation, and info
with st.sidebar:
    st.image("https://th.bing.com/th/id/OIP.D9HbBdI_rOzPJJGR54kLbgHaE7?w=249&h=180&c=7&r=0&o=7&cb=iwp2&dpr=1.3&pid=1.7&rm=3")
    st.title("Magic ML 🚀")
    choice = st.radio("Navigation 🧭", ["Upload 📤", "Profiling 📊", "Modelling 🤖", "Download 💾"])
    st.info("This project application helps you build and explore your data. 📈✨")

def preprocess_data(df, target_column):
    """Preprocess the data for machine learning"""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle categorical variables
    categorical_columns = X.select_dtypes(include=['object']).columns
    le_dict = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
    
    # Handle missing values
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    return X, y, le_dict

def get_models():
    """Return a dictionary of regression models"""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
        'Support Vector Regression': SVR(kernel='rbf')
    }
    return models

def evaluate_models(X, y):
    """Evaluate multiple models and return results"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features for models that benefit from scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = get_models()
    results = []
    
    # Models that need scaling
    scale_models = ['Ridge Regression', 'Lasso Regression', 'K-Nearest Neighbors', 'Support Vector Regression']
    
    for name, model in models.items():
        try:
            # Use scaled data for certain models
            if name in scale_models:
                X_train_use = X_train_scaled
                X_test_use = X_test_scaled
            else:
                X_train_use = X_train
                X_test_use = X_test
            
            # Train model
            model.fit(X_train_use, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_use)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation score
            if name in scale_models:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results.append({
                'Model': name,
                'R²': round(r2, 4),
                'RMSE': round(rmse, 4),
                'MAE': round(mae, 4),
                'CV Mean R²': round(cv_scores.mean(), 4),
                'CV Std R²': round(cv_scores.std(), 4)
            })
            
        except Exception as e:
            st.warning(f"Error training {name}: {str(e)}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('R²', ascending=False)
    
    return results_df, scaler, X_train, X_test, y_train, y_test

# Upload tab: upload CSV file and display dataframe
if choice == "Upload 📤":
    st.title("Upload Your Dataset 📥")
    file = st.file_uploader("Upload Your Dataset 🗂️")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=False)  # Save for later use
        st.dataframe(df)
        st.success(f"Dataset uploaded successfully! Shape: {df.shape}")

# Profiling tab: show detailed EDA report
if choice == "Profiling 📊":
    st.title("Exploratory Data Analysis 🔍")
    if 'df' in locals():
        st.info("Generating comprehensive data profiling report...")
        profile_df = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
        st_profile_report(profile_df)
    else:
        st.warning("⚠️ Please upload a dataset first.")

# Modelling tab: setup scikit-learn, compare models, display results
if choice == "Modelling 🤖":
    st.title("Machine Learning Modelling 🤖")
    if 'df' in locals():
        st.subheader("Model Configuration")
        chosen_target = st.selectbox('Choose the Target Column 🎯', df.columns)
        
        # Show basic info about the dataset
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dataset Shape", f"{df.shape[0]} x {df.shape[1]}")
        with col2:
            st.metric("Target Column", chosen_target)
        with col3:
            st.metric("Features", df.shape[1] - 1)
        
        if st.button('Run Modelling ▶️'):
            with st.spinner('Training models... This may take a few minutes.'):
                try:
                    # Preprocess data
                    X, y, le_dict = preprocess_data(df, chosen_target)
                    
                    # Display preprocessing info
                    st.subheader("Data Preprocessing Summary")
                    st.write(f"- Features shape: {X.shape}")
                    st.write(f"- Target shape: {y.shape}")
                    if le_dict:
                        st.write(f"- Categorical columns encoded: {list(le_dict.keys())}")
                    
                    # Evaluate models
                    results_df, scaler, X_train, X_test, y_train, y_test = evaluate_models(X, y)
                    
                    # Display results
                    st.subheader("Model Comparison Results 📊")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Get best model
                    best_model_name = results_df.iloc[0]['Model']
                    best_r2 = results_df.iloc[0]['R²']
                    
                    st.success(f"🏆 Best Model: {best_model_name} (R² = {best_r2})")
                    
                    # Train and save the best model
                    models = get_models()
                    best_model = models[best_model_name]
                    
                    # Use appropriate data (scaled or not)
                    scale_models = ['Ridge Regression', 'Lasso Regression', 'K-Nearest Neighbors', 'Support Vector Regression']
                    if best_model_name in scale_models:
                        X_train_use = scaler.fit_transform(X_train)
                        best_model.fit(X_train_use, y_train)
                    else:
                        best_model.fit(X_train, y_train)
                    
                    # Save model and preprocessing objects
                    model_data = {
                        'model': best_model,
                        'scaler': scaler if best_model_name in scale_models else None,
                        'label_encoders': le_dict,
                        'feature_names': list(X.columns),
                        'target_name': chosen_target,
                        'model_name': best_model_name,
                        'performance': best_r2
                    }
                    
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(model_data, f)
                    
                    st.session_state['model_trained'] = True
                    st.session_state['model_name'] = best_model_name
                    st.session_state['model_r2'] = best_r2
                    
                except Exception as e:
                    st.error(f"Error during modelling: {str(e)}")
    else:
        st.warning("⚠️ Please upload a dataset first.")

# Download tab: download the saved model file
if choice == "Download 💾":
    st.title("Download Trained Model 💾")
    
    if os.path.exists('best_model.pkl'):
        # Show model info if available
        if 'model_trained' in st.session_state:
            st.success(f"✅ Model Ready: {st.session_state.get('model_name', 'Unknown')}")
            st.info(f"📈 Performance (R²): {st.session_state.get('model_r2', 'Unknown')}")
        
        # Load and display model info
        try:
            with open('best_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            st.subheader("Model Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Model Type:** {model_data.get('model_name', 'Unknown')}")
                st.write(f"**Target Variable:** {model_data.get('target_name', 'Unknown')}")
            with col2:
                st.write(f"**Performance (R²):** {model_data.get('performance', 'Unknown')}")
                st.write(f"**Features:** {len(model_data.get('feature_names', []))}")
        
        except Exception as e:
            st.warning(f"Could not load model info: {str(e)}")
        
        # Download button
        with open('best_model.pkl', 'rb') as f:
            st.download_button(
                label='Download Model 📥',
                data=f.read(),
                file_name="best_model.pkl",
                mime="application/octet-stream"
            )
        
        st.info("💡 The downloaded model includes the trained model, preprocessors, and metadata needed for predictions.")
        
    else:
        st.warning("⚠️ No saved model found. Train a model first.")
