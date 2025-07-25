// API Types and Interfaces for Cancer Alpha Web App

export interface PredictionRequest {
  patient_id: string;
  age: number;
  gender: string;
  features: Record<string, number>;
  model_type?: string;
}

export interface ConfidenceMetrics {
  prediction_confidence: number;
  confidence_level: string;
  entropy: number;
  top_2_margin: number;
}

export interface FeatureExplanation {
  feature_name: string;
  shap_value: number;
  feature_value: number;
  contribution: string;
  importance_rank: number;
}

export interface ExplanationSummary {
  top_positive_features: FeatureExplanation[];
  top_negative_features: FeatureExplanation[];
  explanation_available: boolean;
  explanation_method: string;
  base_value: number;
  prediction_value: number;
}

export interface PredictionResponse {
  patient_id: string;
  predicted_cancer_type: string;
  predicted_cancer_name: string;
  confidence_metrics: ConfidenceMetrics;
  probability_distribution: Record<string, number>;
  explanation: ExplanationSummary;
  model_used: string;
  timestamp: string;
  processing_time_ms: number;
  model_accuracy: number;
  // Legacy field for backward compatibility
  confidence?: number;
}

export interface ModelInfo {
  loaded_models: string[];
  scaler_loaded: boolean;
  feature_importance_loaded: boolean;
  metadata_loaded: boolean;
  model_performance?: Record<string, ModelPerformance>;
  feature_count?: number;
}

export interface ModelPerformance {
  test_accuracy: number;
  cv_mean?: number;
}

export interface HealthStatus {
  status: string;
  timestamp: string;
  models_loaded: boolean;
  message: string;
}

export interface CancerType {
  code: string;
  name: string;
}

export interface CancerTypesResponse {
  cancer_types: string[];
  descriptions: Record<string, string>;
  total_types: number;
}

export interface FeatureImportance {
  top_features: Record<string, number>;
  total_features: number;
  importance_scale: string;
}

export interface Patient {
  id: string;
  name: string;
  age: number;
  gender: string;
  createdAt: string;
  lastPrediction?: PredictionResponse;
}

export interface ApiError {
  detail: string;
  status_code: number;
}
