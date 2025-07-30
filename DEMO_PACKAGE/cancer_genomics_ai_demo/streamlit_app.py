import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
from pathlib import Path

# --- Configuration ---
CANCER_TYPES = ['BRCA', 'LUAD', 'COAD', 'STAD', 'BLCA', 'LIHC', 'CESC', 'KIRP']
N_FEATURES = 110
MODEL_FILENAME = "simple_logistic_regression.pkl"
SCALER_FILENAME = "simple_scaler.pkl"

# --- Model and Scaler Generation ---
def generate_and_save_model_and_scaler(models_dir: Path):
    """Generates and saves a simple logistic regression model and scaler."""
    # Create dummy data for training
    np.random.seed(42)
    X_train = np.random.rand(100, N_FEATURES)
    y_train = np.random.randint(0, len(CANCER_TYPES), 100)

    # Create and fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Create and train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Save scaler and model
    with open(models_dir / SCALER_FILENAME, 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(models_dir / MODEL_FILENAME, 'wb') as f:
        pickle.dump(model, f)
    
    return model, scaler

# --- Main App ---
def main():
    st.title("🌟 Simplified Cancer Genomics AI Demo 🌟")
    
    # --- Setup ---
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / MODEL_FILENAME
    scaler_path = models_dir / SCALER_FILENAME

    # --- Load or Generate Model/Scaler ---
    try:
        if not model_path.exists() or not scaler_path.exists():
            st.warning("Model or scaler not found. Generating new ones...")
            model, scaler = generate_and_save_model_and_scaler(models_dir)
            st.success("✅ New model and scaler generated and saved!")
        else:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            st.success("✅ Model and scaler loaded successfully!")

    except Exception as e:
        st.error(f"❌ Error loading or generating model/scaler: {e}")
        st.info("Attempting to regenerate the model and scaler...")
        try:
            model, scaler = generate_and_save_model_and_scaler(models_dir)
            st.success("✅ New model and scaler generated and saved!")
        except Exception as e2:
            st.error(f"❌ Failed to regenerate model/scaler: {e2}")
            return

    # --- Sidebar ---
    st.sidebar.header("Genomic Profile")
    st.sidebar.info(f"Enter {N_FEATURES} feature values, or use the button below to generate a random sample.")

    if st.sidebar.button("Generate Random Cancer Sample"):
        # Generate a sample with higher values for demonstration
        sample_data = np.random.normal(0.6, 0.3, N_FEATURES)
    else:
        # Create sliders for manual input
        sample_data = np.array([
            st.sidebar.slider(f'Feature {i+1}', -3.0, 3.0, np.random.randn() * 0.5, 0.1) 
            for i in range(5) # Only show first 5 for simplicity
        ] + list(np.random.randn(N_FEATURES - 5) * 0.5))

    # --- Prediction ---
    if 'model' in locals() and 'scaler' in locals():
        # Preprocess and predict
        input_data = sample_data.reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        
        try:
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            confidence = probabilities.max()
            predicted_cancer_type = CANCER_TYPES[prediction]

            # --- Display Results ---
            st.header("Prediction Results")
            st.metric("Predicted Cancer Type", predicted_cancer_type)
            st.metric("Confidence", f"{confidence:.2%}")

            # Show probabilities
            prob_df = pd.DataFrame({
                "Cancer Type": CANCER_TYPES,
                "Probability": probabilities
            }).sort_values("Probability", ascending=False)
            st.dataframe(prob_df)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()

