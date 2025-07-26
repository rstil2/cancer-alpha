#!/usr/bin/env python3
"""
Cancer Genomics Classification Web App with SHAP Explainability
===============================================================

A Streamlit-based interactive web application that loads trained cancer 
classification models, accepts genomic input data, and provides predictions
with confidence scores and SHAP-based explanations.

Features:
- Model loading and selection
- Interactive genomic data input
- Real-time predictions with confidence scores
- SHAP explainability visualizations
- Feature importance analysis
- Biological insights generation

Author: Cancer Alpha Research Team
Date: July 26, 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Cancer Genomics AI Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define feature names matching the trained models (110 features total)
FEATURE_NAMES = (
    # Methylation features (20 features)
    [f'methylation_{i}' for i in range(20)] +
    # Mutation features (25 features)
    [f'mutation_{i}' for i in range(25)] +
    # Copy number alteration features (20 features)
    [f'cn_alteration_{i}' for i in range(20)] +
    # Fragmentomics features (15 features)
    [f'fragmentomics_{i}' for i in range(15)] +
    # Clinical features (10 features)
    [f'clinical_{i}' for i in range(10)] +
    # ICGC ARGO features (20 features)
    [f'icgc_argo_{i}' for i in range(20)]
)

# Generate feature descriptions based on categories
def generate_feature_descriptions():
    descriptions = {}
    
    # Methylation features
    for i in range(20):
        descriptions[f'methylation_{i}'] = f'Methylation pattern feature {i+1} - DNA methylation levels'
    
    # Mutation features
    for i in range(25):
        descriptions[f'mutation_{i}'] = f'Mutation feature {i+1} - Genetic variant information'
    
    # Copy number alteration features
    for i in range(20):
        descriptions[f'cn_alteration_{i}'] = f'Copy number alteration feature {i+1} - Chromosomal gains/losses'
    
    # Fragmentomics features
    for i in range(15):
        descriptions[f'fragmentomics_{i}'] = f'Fragmentomics feature {i+1} - cfDNA fragment characteristics'
    
    # Clinical features
    for i in range(10):
        descriptions[f'clinical_{i}'] = f'Clinical feature {i+1} - Patient clinical information'
    
    # ICGC ARGO features
    for i in range(20):
        descriptions[f'icgc_argo_{i}'] = f'ICGC ARGO feature {i+1} - International cancer genomics data'
    
    return descriptions

FEATURE_DESCRIPTIONS = generate_feature_descriptions()

class CancerClassifierApp:
    """Main application class for the cancer classifier web app"""
    
    def __init__(self):
        self.models_dir = Path("/Users/stillwell/projects/cancer-alpha/models/phase2_models")
        self.models = {}
        self.scalers = {}
        self.feature_names = FEATURE_NAMES
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models and scalers"""
        try:
            # Load available models
            model_files = {
                'Random Forest': 'random_forest_model.pkl',
                'Gradient Boosting': 'gradient_boosting_model.pkl',
                'Deep Neural Network': 'deep_neural_network_model.pkl',
                'Ensemble': 'ensemble_model.pkl'
            }
            
            for model_name, filename in model_files.items():
                model_path = self.models_dir / filename
                if model_path.exists():
                    try:
                        self.models[model_name] = joblib.load(model_path)
                        st.success(f"✅ Loaded {model_name} model")
                    except Exception as e:
                        st.warning(f"⚠️ Could not load {model_name}: {str(e)}")
            
            # Load scaler
            scaler_path = self.models_dir / 'scaler.pkl'
            if scaler_path.exists():
                self.scalers['main'] = joblib.load(scaler_path)
                st.success("✅ Loaded data scaler")
            else:
                st.warning("⚠️ No scaler found - creating dummy scaler")
                from sklearn.preprocessing import StandardScaler
                self.scalers['main'] = StandardScaler()
                
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
    
    def generate_sample_data(self, cancer_type="cancer"):
        """Generate sample data for testing"""
        np.random.seed(42 if cancer_type == "cancer" else 24)
        
        if cancer_type == "cancer":
            # Cancer-like patterns
            data = np.random.normal(0, 1, len(self.feature_names))
            # Adjust specific features for cancer patterns
            data[0] += 0.8   # higher global methylation
            data[9] += 1.2   # more hypermethylation events
            data[10] -= 15   # shorter fragments
            data[12] += 0.3  # more short fragments
            data[21] += 50   # more CNAs
            data[24] += 0.8  # more instability
        else:
            # Control-like patterns
            data = np.random.normal(0, 0.5, len(self.feature_names))
        
        return data
    
    def preprocess_input(self, input_data):
        """Preprocess input data for model prediction"""
        # Convert to numpy array if it's a list
        if isinstance(input_data, list):
            input_data = np.array(input_data)
        
        # Reshape to 2D array for scaler
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        
        # Scale the data
        scaled_data = self.scalers['main'].transform(input_data)
        return scaled_data
    
    def predict_with_confidence(self, model, input_data):
        """Make prediction with confidence scores"""
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        confidence_score = max(probabilities)
        cancer_probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        
        return {
            'prediction': int(prediction),
            'cancer_probability': float(cancer_probability),
            'confidence_score': float(confidence_score),
            'class_probabilities': probabilities.tolist()
        }
    
    def generate_shap_explanation(self, model, input_data, model_name):
        """Generate SHAP explanations for the prediction"""
        try:
            # Create explainer based on model type
            if "Neural Network" in model_name or "Ensemble" in model_name:
                # For neural networks, use a simpler explainer
                explainer = shap.Explainer(model.predict_proba, input_data)
            else:
                # For tree-based models
                explainer = shap.Explainer(model)
            
            # Generate SHAP values
            shap_values = explainer(input_data)
            
            return shap_values
        except Exception as e:
            st.warning(f"Could not generate SHAP explanations for {model_name}: {str(e)}")
            return None
    
    def plot_shap_waterfall(self, shap_values, input_data):
        """Create SHAP waterfall plot"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.plots.waterfall(shap_values[0], show=False)
            plt.tight_layout()
            return fig
        except Exception as e:
            st.warning(f"Could not create waterfall plot: {str(e)}")
            return None
    
    def plot_feature_importance(self, shap_values):
        """Create feature importance plot from SHAP values"""
        try:
            # Extract SHAP values for the positive class
            if hasattr(shap_values, 'values'):
                values = shap_values.values[0]
                if len(values.shape) > 1:
                    values = values[:, 1]  # Take positive class
            else:
                values = shap_values[0]
                if len(values.shape) > 1:
                    values = values[:, 1]
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'SHAP_Value': values,
                'Abs_SHAP_Value': np.abs(values)
            }).sort_values('Abs_SHAP_Value', ascending=True)
            
            # Create horizontal bar plot
            fig = px.bar(
                importance_df.tail(15),
                x='SHAP_Value',
                y='Feature',
                orientation='h',
                color='SHAP_Value',
                color_continuous_scale='RdBu_r',
                title='Top 15 Feature Contributions (SHAP Values)'
            )
            fig.update_layout(height=600)
            return fig
            
        except Exception as e:
            st.warning(f"Could not create feature importance plot: {str(e)}")
            return None
    
    def plot_modality_importance(self, shap_values):
        """Plot importance by modality (methylation, fragmentomics, CNA)"""
        try:
            # Extract SHAP values
            if hasattr(shap_values, 'values'):
                values = shap_values.values[0]
                if len(values.shape) > 1:
                    values = values[:, 1]
            else:
                values = shap_values[0]
                if len(values.shape) > 1:
                    values = values[:, 1]
            
            # Calculate modality contributions
            modality_scores = {
                'Methylation': 0,
                'Mutations': 0,
                'CNA': 0,
                'Fragmentomics': 0,
                'Clinical': 0,
                'ICGC_ARGO': 0
            }
            
            for i, feature in enumerate(self.feature_names):
                abs_shap = abs(values[i])
                if feature.startswith('methylation_'):
                    modality_scores['Methylation'] += abs_shap
                elif feature.startswith('mutation_'):
                    modality_scores['Mutations'] += abs_shap
                elif feature.startswith('cn_alteration_'):
                    modality_scores['CNA'] += abs_shap
                elif feature.startswith('fragmentomics_'):
                    modality_scores['Fragmentomics'] += abs_shap
                elif feature.startswith('clinical_'):
                    modality_scores['Clinical'] += abs_shap
                elif feature.startswith('icgc_argo_'):
                    modality_scores['ICGC_ARGO'] += abs_shap
            
            # Create pie chart
            fig = px.pie(
                values=list(modality_scores.values()),
                names=list(modality_scores.keys()),
                title='Feature Importance by Genomic Modality'
            )
            return fig, modality_scores
            
        except Exception as e:
            st.warning(f"Could not create modality plot: {str(e)}")
            return None, {}

def main():
    """Main Streamlit application"""
    
    # App header
    st.title("🧬 Cancer Genomics AI Classifier")
    st.markdown("""
    This interactive web application uses trained machine learning models to classify cancer 
    from multi-modal genomic data including methylation patterns, fragmentomics profiles, 
    and copy number alterations. The app provides predictions with confidence scores and 
    SHAP-based explanations for model interpretability.
    """)
    
    # Initialize app
    app = CancerClassifierApp()
    
    # Sidebar for model selection and input options
    st.sidebar.header("🔧 Configuration")
    
    # Model selection
    if app.models:
        selected_model = st.sidebar.selectbox(
            "Select Model",
            list(app.models.keys()),
            help="Choose which trained model to use for predictions"
        )
    else:
        st.error("❌ No models loaded. Please check model files.")
        return
    
    # Input method selection
    input_method = st.sidebar.radio(
        "Input Method",
        ["Sample Data", "Manual Input", "Upload CSV"],
        help="Choose how to provide genomic data"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Input Data")
        
        input_data = None
        
        if input_method == "Sample Data":
            sample_type = st.selectbox(
                "Sample Type",
                ["Cancer Sample", "Control Sample"],
                help="Generate representative sample data"
            )
            
            if st.button("Generate Sample Data"):
                sample_data_type = "cancer" if sample_type == "Cancer Sample" else "control"
                input_data = app.generate_sample_data(sample_data_type)
                
                # Display data
                df = pd.DataFrame({
                    'Feature': app.feature_names,
                    'Value': input_data,
                    'Description': [FEATURE_DESCRIPTIONS.get(f, 'N/A') for f in app.feature_names]
                })
                st.dataframe(df, height=400)
        
        elif input_method == "Manual Input":
            st.markdown("Adjust feature values manually:")
            
            # Create input widgets for each feature
            input_values = []
            for i, feature in enumerate(app.feature_names):
                default_val = 0.0
                description = FEATURE_DESCRIPTIONS.get(feature, feature)
                
                # Determine reasonable ranges based on feature type
                if 'ratio' in feature or 'proportion' in feature:
                    min_val, max_val = 0.0, 1.0
                elif 'length' in feature:
                    min_val, max_val = 50.0, 200.0
                elif 'alterations' in feature or 'events' in feature:
                    min_val, max_val = 0.0, 100.0
                else:
                    min_val, max_val = -3.0, 3.0
                
                value = st.number_input(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    step=0.1,
                    help=description,
                    key=f"input_{i}"
                )
                input_values.append(value)
            
            input_data = np.array(input_values)
        
        elif input_method == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=['csv'],
                help="Upload a CSV file with genomic features"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.dataframe(df.head())
                    
                    if len(df.columns) == len(app.feature_names):
                        input_data = df.iloc[0].values
                        st.success("✅ Data loaded successfully")
                    else:
                        st.error(f"❌ Expected {len(app.feature_names)} features, got {len(df.columns)}")
                except Exception as e:
                    st.error(f"❌ Error reading CSV: {str(e)}")
    
    with col2:
        st.header("🤖 Model Info")
        st.info(f"**Selected Model:** {selected_model}")
        
        if app.models:
            model = app.models[selected_model]
            st.write(f"**Model Type:** {type(model).__name__}")
            
            # Model performance (if available)
            st.write("**Features:** Multi-modal genomic data")
            st.write("**Classes:** Cancer vs Control")
    
    # Prediction section
    if input_data is not None:
        st.header("🔮 Prediction Results")
        
        try:
            # Preprocess input
            processed_data = app.preprocess_input(input_data)
            
            # Make prediction
            model = app.models[selected_model]
            prediction_result = app.predict_with_confidence(model, processed_data)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                prediction_text = "🔴 CANCER DETECTED" if prediction_result['prediction'] == 1 else "🟢 NO CANCER"
                st.metric("Prediction", prediction_text)
            
            with col2:
                st.metric("Cancer Probability", f"{prediction_result['cancer_probability']:.1%}")
            
            with col3:
                st.metric("Confidence Score", f"{prediction_result['confidence_score']:.1%}")
            
            # Confidence visualization
            prob_df = pd.DataFrame({
                'Class': ['Control', 'Cancer'],
                'Probability': prediction_result['class_probabilities']
            })
            
            fig_prob = px.bar(
                prob_df,
                x='Class',
                y='Probability',
                color='Class',
                color_discrete_map={'Cancer': 'red', 'Control': 'green'},
                title='Class Probabilities'
            )
            st.plotly_chart(fig_prob, use_container_width=True)
            
            # SHAP Explanations
            st.header("🔍 Model Explanations (SHAP)")
            
            with st.spinner("Generating SHAP explanations..."):
                shap_values = app.generate_shap_explanation(model, processed_data, selected_model)
                
                if shap_values is not None:
                    # Feature importance plot
                    fig_importance = app.plot_feature_importance(shap_values)
                    if fig_importance:
                        st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Modality importance
                    fig_modality, modality_scores = app.plot_modality_importance(shap_values)
                    if fig_modality:
                        st.plotly_chart(fig_modality, use_container_width=True)
                    
                    # Biological insights
                    st.header("🧪 Biological Insights")
                    
                    if modality_scores:
                        top_modality = max(modality_scores, key=modality_scores.get)
                        
                        insights = []
                        if modality_scores.get('Methylation', 0) > 0.3:
                            insights.append("🔬 **Methylation patterns** show significant contribution - suggests epigenetic alterations characteristic of cancer")
                        
                        if modality_scores.get('Fragmentomics', 0) > 0.25:
                            insights.append("🧬 **Fragmentomics profile** indicates altered nucleosome positioning - potential for non-invasive liquid biopsy")
                        
                        if modality_scores.get('CNA', 0) > 0.3:
                            insights.append("📊 **Copy number alterations** suggest genomic instability - correlates with tumor progression")
                        
                        if prediction_result['cancer_probability'] > 0.7:
                            insights.append("⚠️ **High cancer probability** - recommend further clinical evaluation")
                        
                        for insight in insights:
                            st.markdown(insight)
                        
                        if not insights:
                            st.info("Model prediction is based on subtle pattern combinations across multiple genomic modalities.")
                else:
                    st.warning("SHAP explanations not available for this prediction.")
                    
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()
