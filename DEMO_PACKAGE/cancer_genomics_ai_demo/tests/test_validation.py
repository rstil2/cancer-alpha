import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.multimodal_transformer import MultiModalTransformer, MultiModalConfig
import joblib

# Load model
def load_model():
    checkpoint = torch.load('models/optimized_multimodal_transformer.pth', weights_only=False)
    config = checkpoint.get('config', MultiModalConfig())
    model = MultiModalTransformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Load scalers
scalers = joblib.load('models/scalers.pkl')

# Test cases

def test_synthetic_samples(model):
    """Tests predictions on synthetic data samples"""
    # Generate synthetic test data
    synthetic_data = np.random.randn(5, 110)
    
    # Scale the data
    methylation = scalers['methylation'].transform(synthetic_data[:, :20])
    mutation = scalers['mutation'].transform(synthetic_data[:, 20:45])
    cna = scalers['cna'].transform(synthetic_data[:, 45:65])
    fragmentomics = scalers['fragmentomics'].transform(synthetic_data[:, 65:80])
    clinical = scalers['clinical'].transform(synthetic_data[:, 80:90])
    icgc = scalers['icgc'].transform(synthetic_data[:, 90:110])

    scaled_data = np.concatenate([methylation, mutation, cna, fragmentomics, clinical, icgc], axis=1)

    # Prepare data for the transformer
    data_dict = {
        'methylation': torch.FloatTensor(scaled_data[:, :20]),
        'mutation': torch.FloatTensor(scaled_data[:, 20:45]),
        'cna': torch.FloatTensor(scaled_data[:, 45:65]),
        'fragmentomics': torch.FloatTensor(scaled_data[:, 65:80]),
        'clinical': torch.FloatTensor(scaled_data[:, 80:90]),
        'icgc': torch.FloatTensor(scaled_data[:, 90:110])
    }
    
    # Predictions
    with torch.no_grad():
        output = model(data_dict)
        probabilities = output['probabilities'].numpy()
        predictions = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)
        
    # Log results
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        print(f"Sample {i}: Predicted class = {pred} with confidence = {conf:.2f}")


if __name__ == '__main__':
    model = load_model()
    test_synthetic_samples(model)

