import axios from 'axios';
import {
  PredictionRequest,
  PredictionResponse,
  ModelInfo,
  HealthStatus,
  CancerTypesResponse,
  FeatureImportance,
  ApiError,
} from '../types/api';

// Configure axios defaults
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for adding auth headers if needed
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for handling errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const apiError: ApiError = {
      detail: error.response?.data?.detail || error.message || 'An error occurred',
      status_code: error.response?.status || 500,
    };
    return Promise.reject(apiError);
  }
);

// API service functions
export const apiService = {
  // Health check
  async checkHealth(): Promise<HealthStatus> {
    const response = await api.get('/health');
    return response.data;
  },

  // Get API root info
  async getApiInfo(): Promise<any> {
    const response = await api.get('/');
    return response.data;
  },

  // Get model information
  async getModelInfo(): Promise<ModelInfo> {
    const response = await api.get('/models/info');
    return response.data;
  },

  // Get cancer types
  async getCancerTypes(): Promise<CancerTypesResponse> {
    const response = await api.get('/cancer-types');
    return response.data;
  },

  // Get feature importance
  async getFeatureImportance(): Promise<FeatureImportance> {
    const response = await api.get('/models/feature-importance');
    return response.data;
  },

  // Get feature information and explainability details
  async getFeatureInfo(): Promise<any> {
    const response = await api.get('/features/info');
    return response.data;
  },

  // Make prediction
  async makePrediction(request: PredictionRequest): Promise<PredictionResponse> {
    const response = await api.post('/predict', request);
    return response.data;
  },

  // Generate sample features for testing
  generateSampleFeatures(): Record<string, number> {
    const features: Record<string, number> = {};
    
    // Generate 110 random features as expected by the model
    for (let i = 0; i < 110; i++) {
      features[`feature_${i}`] = Math.random() * 2 - 1; // Random values between -1 and 1
    }
    
    return features;
  },

  // Validate feature data
  validateFeatures(features: Record<string, number>): boolean {
    const expectedFeatureCount = 110;
    const featureKeys = Object.keys(features);
    
    // Check if we have the right number of features
    if (featureKeys.length !== expectedFeatureCount) {
      return false;
    }
    
    // Check if all features are valid numbers
    for (const key of featureKeys) {
      if (typeof features[key] !== 'number' || isNaN(features[key])) {
        return false;
      }
    }
    
    return true;
  },

  // Parse CSV data to features
  parseCSVToFeatures(csvData: string): Record<string, number> | null {
    try {
      const lines = csvData.trim().split('\n');
      const headers = lines[0].split(',');
      
      if (lines.length < 2) {
        throw new Error('CSV must have at least 2 lines (header + data)');
      }
      
      const dataLine = lines[1].split(',');
      const features: Record<string, number> = {};
      
      for (let i = 0; i < headers.length; i++) {
        const header = headers[i].trim();
        const value = parseFloat(dataLine[i]);
        
        if (!isNaN(value)) {
          features[header] = value;
        }
      }
      
      return features;
    } catch (error) {
      console.error('Error parsing CSV:', error);
      return null;
    }
  },
};

export default apiService;
