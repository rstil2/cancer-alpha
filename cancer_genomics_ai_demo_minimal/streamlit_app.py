#!/usr/bin/env python3
"""
Oncura Cancer Genomics AI Classifier — DEMONSTRATION VERSION
============================================================

Research demo only. Not for clinical diagnosis or treatment decisions.

FULL SYSTEM (Study 2, separate pipeline — not this demo):
- 98.4% balanced accuracy on held-out TCGA test (LightGBM, n=1,248)
- Reproduce via src/pipeline/ — see RESEARCH.md

THIS DEMO:
- Logistic Regression and Random Forest on 158 TCGA samples, 110 features
- Sample/illustrative data in UI (~70% accuracy on demo inputs)

Author: Dr. R. Craig Stillwell
License: Academic use — see LICENSE
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# SHAP is optional — graceful fallback if unavailable
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Import usage tracker for demo monitoring
try:
    from usage_tracker import demo_tracker
except ImportError:
    # Create dummy tracker if import fails
    class DummyTracker:
        def log_demo_start(self): pass
        def log_prediction_made(self, *args): pass
        def log_demo_end(self): pass
        def get_usage_summary(self): return {"total_sessions": 0, "total_predictions": 0}
    demo_tracker = DummyTracker()

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
        # Use local models directory for demo
        self.models_dir = Path(__file__).parent / "models"
        self.models = {}
        self.scalers = {}
        self.feature_names = FEATURE_NAMES

        # Define cancer types matching the demo models (minimal-data setting)
        self.cancer_types = ['BRCA', 'LUAD', 'COAD', 'PRAD', 'STAD', 'KIRC', 'HNSC', 'LIHC']
        self.load_models()
    
    def load_models(self):
        """Load pre-trained demo models and scalers"""
        try:
            # Load available demo models
            model_files = {
                'Random Forest (Demo)': 'multimodal_real_tcga_random_forest.pkl',
                'Logistic Regression (Demo)': 'multimodal_real_tcga_logistic_regression.pkl',
            }
            
            for model_name, filename in model_files.items():
                model_path = self.models_dir / filename
                if model_path.exists():
                    try:
                        self.models[model_name] = joblib.load(model_path)
                    except Exception as e:
                        st.warning(f"⚠️ Could not load {model_name}: {str(e)}")
            
            if self.models:
                st.success(f"✅ Loaded {len(self.models)} demo model(s)")
            else:
                st.warning("⚠️ No demo models found")
            
            # Load scalers
            scalers_path = self.models_dir / 'scalers.pkl'
            if scalers_path.exists():
                scalers_data = joblib.load(scalers_path)
                if isinstance(scalers_data, dict):
                    self.scalers = scalers_data
                else:
                    self.scalers['main'] = scalers_data
                
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
    
    def generate_sample_data(self, cancer_type="cancer"):
        """Generate sample data that matches the training data patterns for realistic demo"""
        np.random.seed(42 if cancer_type == "cancer" else 24)
        
        if cancer_type == "cancer":
            # Generate BRCA-like cancer sample matching the training data structure
            sample_data = []
            
            # Base pattern for BRCA (class 0) from training
            base_pattern = 0.2
            
            # Methylation features (20) - higher methylation typical of cancer
            sample_data.extend(np.random.normal(base_pattern + 0.3, 0.1, 20))
            
            # Mutation features (25) - more mutations in cancer
            sample_data.extend(np.random.poisson(5, 25))
            
            # Copy number alteration features (20) - more CNAs in cancer
            sample_data.extend(np.random.normal(10, 2, 20))
            
            # Fragmentomics features (15) - shorter fragments in cancer
            sample_data.extend(np.random.exponential(150, 15))
            
            # Clinical features (10) - cancer-associated values
            sample_data.extend(np.random.normal(0.5, 0.1, 10))
            
            # ICGC ARGO features (20) - elevated in cancer
            sample_data.extend(np.random.gamma(2, 0.5, 20))
            
            data = np.array(sample_data)
            
        else:
            # Generate clearly healthy control sample with very different patterns
            sample_data = []
            
            # Methylation features (20) - much lower methylation (healthy pattern)
            sample_data.extend(np.random.normal(-0.2, 0.03, 20))
            
            # Mutation features (25) - very few mutations (healthy)
            sample_data.extend(np.random.poisson(0.5, 25))
            
            # Copy number alteration features (20) - minimal alterations
            sample_data.extend(np.random.normal(0, 0.5, 20))
            
            # Fragmentomics features (15) - longer, healthier fragments
            sample_data.extend(np.random.exponential(200, 15))
            
            # Clinical features (10) - healthy values
            sample_data.extend(np.random.normal(-0.3, 0.03, 10))
            
            # ICGC ARGO features (20) - low, healthy levels
            sample_data.extend(np.random.gamma(0.8, 0.2, 20))
            
            data = np.array(sample_data)
        
        return data
    
    # Modality slices for per-modality scaling (matches training pipeline)
    MODALITY_SLICES = [
        ('methylation', 0, 20),
        ('mutation', 20, 45),
        ('cna', 45, 65),
        ('fragmentomics', 65, 80),
        ('clinical', 80, 90),
        ('icgc', 90, 110),
    ]

    def preprocess_input(self, input_data, model_name=None):
        """Preprocess input data for model prediction"""
        if isinstance(input_data, list):
            input_data = np.array(input_data)
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        
        # Apply per-modality scaling
        if self.scalers:
            scaled_parts = []
            for key, start, end in self.MODALITY_SLICES:
                if key in self.scalers:
                    scaled_parts.append(self.scalers[key].transform(input_data[:, start:end]))
                else:
                    scaled_parts.append(input_data[:, start:end])
            scaled_data = np.hstack(scaled_parts)
        else:
            scaled_data = input_data
        
        # Apply feature selection for logistic regression
        if model_name and 'Logistic Regression' in model_name:
            feature_selector_path = self.models_dir / "feature_selector.pkl"
            if feature_selector_path.exists():
                try:
                    feature_selector = joblib.load(feature_selector_path)
                    scaled_data = feature_selector.transform(scaled_data)
                except Exception as e:
                    st.warning(f"Could not apply feature selection: {e}")
            
        return scaled_data
    
    def predict_with_confidence(self, model, input_data):
        """Make prediction with confidence scores for multi-class cancer classification"""
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Normalize probabilities (handles sklearn cross-version compat issues)
        prob_sum = probabilities.sum()
        if prob_sum > 0 and abs(prob_sum - 1.0) > 0.01:
            probabilities = probabilities / prob_sum
        
        confidence_score = max(probabilities)
        predicted_cancer_type = self.cancer_types[prediction]
        
        return {
            'prediction': int(prediction),
            'predicted_cancer_type': predicted_cancer_type,
            'confidence_score': float(confidence_score),
            'class_probabilities': probabilities.tolist(),
            'cancer_types': self.cancer_types
        }
    
    def generate_shap_explanation(self, model, input_data, model_name):
        """Generate SHAP explanations for the prediction"""
        if not SHAP_AVAILABLE:
            return None
        try:
            if "Random Forest" in model_name:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_data)
                if isinstance(shap_values, list) and len(shap_values) > 1:
                    predicted_class = model.predict(input_data)[0]
                    shap_values = shap_values[predicted_class]
                return shap_values
            elif "Logistic Regression" in model_name:
                explainer = shap.LinearExplainer(model, input_data)
                shap_values = explainer.shap_values(input_data)
                if isinstance(shap_values, list) and len(shap_values) > 1:
                    predicted_class = model.predict(input_data)[0]
                    shap_values = shap_values[predicted_class]
                return shap_values
            else:
                explainer = shap.Explainer(model.predict_proba, input_data)
                return explainer(input_data)
        except Exception as e:
            st.warning(f"Could not generate SHAP explanations: {str(e)}")
            return None
    
    def plot_feature_importance(self, shap_values):
        """Create feature importance plot from SHAP values"""
        try:
            # Extract SHAP values - handle different formats
            if hasattr(shap_values, 'values'):
                # New SHAP format
                values = shap_values.values[0]
                if len(values.shape) > 1:
                    values = values[:, 1]  # Take positive class
            elif isinstance(shap_values, np.ndarray):
                # Direct numpy array from TreeExplainer
                values = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            else:
                # List format or other
                values = shap_values[0] if isinstance(shap_values, list) else shap_values
                if hasattr(values, 'shape') and len(values.shape) > 1:
                    values = values[:, 1]
            
            # Flatten if needed
            if len(values.shape) > 1:
                values = values.flatten()
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'Feature': self.feature_names[:len(values)],  # Match feature length
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
            # Extract SHAP values - handle different formats
            if hasattr(shap_values, 'values'):
                # New SHAP format
                values = shap_values.values[0]
                if len(values.shape) > 1:
                    values = values[:, 1]  # Take positive class
            elif isinstance(shap_values, np.ndarray):
                # Direct numpy array from TreeExplainer
                values = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            else:
                # List format or other
                values = shap_values[0] if isinstance(shap_values, list) else shap_values
                if hasattr(values, 'shape') and len(values.shape) > 1:
                    values = values[:, 1]
            
            # Flatten if needed
            if len(values.shape) > 1:
                values = values.flatten()
            
            # Calculate modality contributions
            modality_scores = {
                'Methylation': 0,
                'Mutations': 0,
                'CNA': 0,
                'Fragmentomics': 0,
                'Clinical': 0,
                'ICGC_ARGO': 0
            }
            
            # Only iterate over available values
            for i in range(min(len(values), len(self.feature_names))):
                feature = self.feature_names[i]
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
    
    # Log demo start
    demo_tracker.log_demo_start()
    
    # App header
    st.title("🧬 Oncura — Cancer Genomics AI Classifier (Demo)")
    st.markdown("""
    This interactive demo classifies **cancer types** from multi-modal genomic data
    using machine learning models trained on real TCGA patient samples.
    Predictions include confidence scores and SHAP-based explanations.
    """)
    
    # Full system metrics
    st.success("""
    **Full research pipeline** (not included in this demo — see [RESEARCH.md](../RESEARCH.md)):
    - **98.4%** held-out test balanced accuracy (LightGBM, Study 2, n=1,248)
    - **95.0%** with LightGBM+SMOTE (Study 1, n=158)
    - Reproduce Study 2: `python src/pipeline/step4_train_evaluate.py`
    """)
    
    st.info("""
    📝 **Demo Note**: This demo uses simplified models (Logistic Regression & Random Forest)
    trained on 158 TCGA samples with 110 features. Try "Cancer Sample" and "Control Sample"
    to see how different genomic patterns lead to different cancer type predictions.
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
            st.write("**Classes:** 8 Cancer Types")
            st.write("**Cancer Types:**")
            for i, cancer_type in enumerate(app.cancer_types):
                st.write(f"- {cancer_type}")
    
    # Prediction section
    if input_data is not None:
        st.header("🔮 Prediction Results")
        
        try:
            # Preprocess input with appropriate scaler for selected model
            processed_data = app.preprocess_input(input_data, selected_model)
            
            # Make prediction
            model = app.models[selected_model]
            prediction_result = app.predict_with_confidence(model, processed_data)
            
            # Log prediction for tracking
            demo_tracker.log_prediction_made(
                selected_model, 
                sample_type if input_method == "Sample Data" else input_method,
                prediction_result['confidence_score']
            )
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                predicted_type = prediction_result['predicted_cancer_type']
                prediction_text = f"🔴 {predicted_type} CANCER"
                st.metric("Predicted Cancer Type", predicted_type)
            
            with col2:
                st.metric("Confidence Score", f"{prediction_result['confidence_score']:.1%}")
            
            with col3:
                # Show probability of the predicted class
                predicted_prob = prediction_result['class_probabilities'][prediction_result['prediction']]
                st.metric("Prediction Probability", f"{predicted_prob:.1%}")
            
            # Cancer type probabilities visualization
            prob_df = pd.DataFrame({
                'Cancer_Type': prediction_result['cancer_types'],
                'Probability': prediction_result['class_probabilities']
            })
            
            # Sort by probability for better visualization
            prob_df = prob_df.sort_values('Probability', ascending=True)
            
            fig_prob = px.bar(
                prob_df,
                x='Probability',
                y='Cancer_Type',
                orientation='h',
                color='Probability',
                color_continuous_scale='Reds',
                title='Cancer Type Classification Probabilities'
            )
            fig_prob.update_layout(height=400)
            st.plotly_chart(fig_prob, use_container_width=True)
            
            # SHAP Explanations
            if SHAP_AVAILABLE:
                st.header("🔍 Model Explanations (SHAP)")
            else:
                st.header("🔍 Model Explanations")
                st.warning("SHAP library not available. Install with: `pip install 'shap>=0.42' 'numpy<2.3'`")
            
            if SHAP_AVAILABLE:
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
                            insights = []
                            if modality_scores.get('Methylation', 0) > 0.3:
                                insights.append("🔬 **Methylation patterns** show significant contribution - suggests epigenetic alterations")
                            
                            if modality_scores.get('Fragmentomics', 0) > 0.25:
                                insights.append("🧬 **Fragmentomics profile** indicates altered nucleosome positioning")
                            
                            if modality_scores.get('CNA', 0) > 0.3:
                                insights.append("📊 **Copy number alterations** suggest genomic instability")
                            
                            if prediction_result['confidence_score'] > 0.7:
                                insights.append(f"⚠️ **High confidence {predicted_type} prediction**")
                            
                            for insight in insights:
                                st.markdown(insight)
                            
                            if not insights:
                                st.info("Prediction based on subtle pattern combinations across multiple genomic modalities.")
                    else:
                        st.warning("SHAP explanations not available for this prediction.")
                    
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()
