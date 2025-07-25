import React, { useState, useCallback } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  CircularProgress,
  Chip,
  Divider,
  Paper,
  LinearProgress,
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import {
  Upload,
  Send,
  Analytics,
  Person,
  DataObject,
} from '@mui/icons-material';
import { apiService } from '../services/api';
import { PredictionRequest, PredictionResponse } from '../types/api';

const PredictionInterface: React.FC = () => {
  const [patientData, setPatientData] = useState({
    patient_id: '',
    age: '',
    gender: '',
    model_type: 'ensemble',
  });
  const [features, setFeatures] = useState<Record<string, number>>({});
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setUploadedFile(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        const csvData = e.target?.result as string;
        const parsedFeatures = apiService.parseCSVToFeatures(csvData);
        if (parsedFeatures) {
          setFeatures(parsedFeatures);
          setError(null);
        } else {
          setError('Failed to parse CSV file. Please ensure it has proper format.');
        }
      };
      reader.readAsText(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.csv'],
    },
    maxFiles: 1,
  });

  const handleInputChange = (field: string, value: string) => {
    setPatientData(prev => ({ ...prev, [field]: value }));
  };

  const generateSampleData = () => {
    const sampleFeatures = apiService.generateSampleFeatures();
    setFeatures(sampleFeatures);
    setPatientData(prev => ({
      ...prev,
      patient_id: prev.patient_id || 'SAMPLE_' + Date.now(),
      age: prev.age || '65',
      gender: prev.gender || 'M',
    }));
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!patientData.patient_id || !patientData.age || !patientData.gender) {
      setError('Please fill in all patient information fields');
      return;
    }

    if (Object.keys(features).length === 0) {
      setError('Please upload feature data or generate sample data');
      return;
    }

    if (!apiService.validateFeatures(features)) {
      setError('Invalid feature data. Expected 110 numerical features.');
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const request: PredictionRequest = {
        patient_id: patientData.patient_id,
        age: parseInt(patientData.age),
        gender: patientData.gender,
        features: features,
        model_type: patientData.model_type,
      };

      const result = await apiService.makePrediction(request);
      setPrediction(result);
    } catch (err: any) {
      setError(err.detail || 'Failed to make prediction');
    } finally {
      setLoading(false);
    }
  };

  const clearForm = () => {
    setPatientData({
      patient_id: '',
      age: '',
      gender: '',
      model_type: 'ensemble',
    });
    setFeatures({});
    setPrediction(null);
    setError(null);
    setUploadedFile(null);
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Cancer Prediction Interface
      </Typography>

      <Grid container spacing={3}>
        {/* Patient Information */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Person color="primary" />
                <Typography variant="h6" component="h2" sx={{ ml: 1 }}>
                  Patient Information
                </Typography>
              </Box>
              <Box component="form" onSubmit={handleSubmit}>
                <TextField
                  fullWidth
                  label="Patient ID"
                  value={patientData.patient_id}
                  onChange={(e) => handleInputChange('patient_id', e.target.value)}
                  margin="normal"
                  required
                />
                <TextField
                  fullWidth
                  label="Age"
                  type="number"
                  value={patientData.age}
                  onChange={(e) => handleInputChange('age', e.target.value)}
                  margin="normal"
                  required
                  inputProps={{ min: 0, max: 150 }}
                />
                <FormControl fullWidth margin="normal">
                  <InputLabel>Gender</InputLabel>
                  <Select
                    value={patientData.gender}
                    onChange={(e) => handleInputChange('gender', e.target.value)}
                    label="Gender"
                    required
                  >
                    <MenuItem value="M">Male</MenuItem>
                    <MenuItem value="F">Female</MenuItem>
                  </Select>
                </FormControl>
                <FormControl fullWidth margin="normal">
                  <InputLabel>Model Type</InputLabel>
                  <Select
                    value={patientData.model_type}
                    onChange={(e) => handleInputChange('model_type', e.target.value)}
                    label="Model Type"
                  >
                    <MenuItem value="ensemble">Ensemble (Recommended)</MenuItem>
                    <MenuItem value="random_forest">Random Forest</MenuItem>
                    <MenuItem value="gradient_boosting">Gradient Boosting</MenuItem>
                    <MenuItem value="deep_neural_network">Deep Neural Network</MenuItem>
                  </Select>
                </FormControl>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Data Upload */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <DataObject color="secondary" />
                <Typography variant="h6" component="h2" sx={{ ml: 1 }}>
                  Feature Data
                </Typography>
              </Box>
              
              <Paper
                {...getRootProps()}
                sx={{
                  p: 3,
                  border: '2px dashed #ccc',
                  borderRadius: 2,
                  textAlign: 'center',
                  cursor: 'pointer',
                  backgroundColor: isDragActive ? '#f5f5f5' : 'transparent',
                  mb: 2,
                }}
              >
                <input {...getInputProps()} />
                <Upload color="action" sx={{ fontSize: 48, mb: 1 }} />
                <Typography variant="body1" gutterBottom>
                  {isDragActive
                    ? 'Drop the CSV file here...'
                    : 'Drag and drop a CSV file here, or click to select'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Expected: CSV file with 110 numerical features
                </Typography>
              </Paper>

              {uploadedFile && (
                <Alert severity="success" sx={{ mb: 2 }}>
                  Uploaded: {uploadedFile.name}
                </Alert>
              )}

              <Box display="flex" gap={2} mb={2}>
                <Button
                  variant="outlined"
                  onClick={generateSampleData}
                  startIcon={<Analytics />}
                  fullWidth
                >
                  Generate Sample Data
                </Button>
              </Box>

              {Object.keys(features).length > 0 && (
                <Box>
                  <Typography variant="body2" gutterBottom>
                    Features loaded: {Object.keys(features).length}
                  </Typography>
                  <Chip
                    label={`${Object.keys(features).length} features`}
                    size="small"
                    color="success"
                  />
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Submit Button */}
        <Grid item xs={12}>
          <Box display="flex" gap={2} justifyContent="center">
            <Button
              type="submit"
              variant="contained"
              size="large"
              onClick={handleSubmit}
              disabled={loading}
              startIcon={loading ? <CircularProgress size={20} /> : <Send />}
            >
              {loading ? 'Making Prediction...' : 'Make Prediction'}
            </Button>
            <Button
              variant="outlined"
              size="large"
              onClick={clearForm}
              disabled={loading}
            >
              Clear Form
            </Button>
          </Box>
        </Grid>

        {/* Error Display */}
        {error && (
          <Grid item xs={12}>
            <Alert severity="error">{error}</Alert>
          </Grid>
        )}

        {/* Prediction Results */}
        {prediction && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" component="h2" gutterBottom>
                  Prediction Results
                </Typography>
                <Divider sx={{ mb: 2 }} />
                
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Box>
                      <Typography variant="subtitle1" fontWeight="bold">
                        Predicted Cancer Type
                      </Typography>
                      <Typography variant="h5" color="primary" gutterBottom>
                        {prediction.predicted_cancer_type}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        {prediction.predicted_cancer_name}
                      </Typography>
                      
                      <Typography variant="subtitle1" fontWeight="bold" sx={{ mt: 2 }}>
                        Confidence Metrics
                      </Typography>
                      <Box display="flex" alignItems="center" gap={1} mb={1}>
                        <LinearProgress
                          variant="determinate"
                          value={(prediction.confidence_metrics?.prediction_confidence || prediction.confidence || 0) * 100}
                          sx={{ flexGrow: 1, height: 10, borderRadius: 5 }}
                        />
                        <Typography variant="body2" fontWeight="bold">
                          {((prediction.confidence_metrics?.prediction_confidence || prediction.confidence || 0) * 100).toFixed(1)}%
                        </Typography>
                      </Box>
                      {prediction.confidence_metrics && (
                        <Box>
                          <Chip
                            label={`${prediction.confidence_metrics.confidence_level} Confidence`}
                            size="small"
                            color={prediction.confidence_metrics.confidence_level === 'High' ? 'success' : 
                                   prediction.confidence_metrics.confidence_level === 'Medium' ? 'warning' : 'error'}
                            sx={{ mr: 1, mb: 1 }}
                          />
                          <Typography variant="caption" display="block" color="text.secondary">
                            Entropy: {prediction.confidence_metrics.entropy.toFixed(3)} | 
                            Top 2 Margin: {(prediction.confidence_metrics.top_2_margin * 100).toFixed(1)}%
                          </Typography>
                        </Box>
                      )}
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                      Probability Distribution
                    </Typography>
                    <Box>
                      {Object.entries(prediction.probability_distribution)
                        .sort(([, a], [, b]) => b - a)
                        .slice(0, 5)
                        .map(([cancerType, prob]) => (
                          <Box key={cancerType} sx={{ mb: 1 }}>
                            <Box display="flex" justifyContent="space-between" mb={0.5}>
                              <Typography variant="body2">{cancerType}</Typography>
                              <Typography variant="body2" fontWeight="bold">
                                {(prob * 100).toFixed(1)}%
                              </Typography>
                            </Box>
                            <LinearProgress
                              variant="determinate"
                              value={prob * 100}
                              sx={{ height: 6, borderRadius: 3 }}
                            />
                          </Box>
                        ))}
                    </Box>
                  </Grid>
                </Grid>
                
                {/* SHAP Explainability Section */}
                {prediction.explanation && prediction.explanation.explanation_available && (
                  <>
                    <Divider sx={{ my: 2 }} />
                    <Typography variant="h6" component="h3" gutterBottom>
                      🔍 Feature Explanation (SHAP Analysis)
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom sx={{ mb: 2 }}>
                      Method: {prediction.explanation.explanation_method} | 
                      Base Value: {prediction.explanation.base_value.toFixed(3)}
                    </Typography>
                    
                    <Grid container spacing={2}>
                      {/* Top Positive Features */}
                      <Grid item xs={12} md={6}>
                        <Typography variant="subtitle1" fontWeight="bold" gutterBottom color="success.main">
                          ➕ Top Contributing Features
                        </Typography>
                        {prediction.explanation.top_positive_features
                          .slice(0, 5)
                          .map((feature, index) => (
                            <Box key={feature.feature_name} sx={{ mb: 1.5 }}>
                              <Box display="flex" justifyContent="space-between" alignItems="center" mb={0.5}>
                                <Typography variant="body2" fontWeight="medium">
                                  {feature.feature_name}
                                </Typography>
                                <Box display="flex" alignItems="center" gap={1}>
                                  <Chip
                                    label={`#${feature.importance_rank}`}
                                    size="small"
                                    variant="outlined"
                                    color="primary"
                                  />
                                  <Typography variant="caption" fontWeight="bold" color="success.main">
                                    +{feature.shap_value.toFixed(3)}
                                  </Typography>
                                </Box>
                              </Box>
                              <Box display="flex" alignItems="center" gap={1}>
                                <LinearProgress
                                  variant="determinate"
                                  value={Math.min(Math.abs(feature.shap_value) * 1000, 100)}
                                  sx={{ 
                                    flexGrow: 1, 
                                    height: 6, 
                                    borderRadius: 3,
                                    '& .MuiLinearProgress-bar': {
                                      backgroundColor: '#4caf50'
                                    }
                                  }}
                                />
                                <Typography variant="caption" color="text.secondary">
                                  Value: {feature.feature_value.toFixed(2)}
                                </Typography>
                              </Box>
                            </Box>
                          ))}
                      </Grid>
                      
                      {/* Top Negative Features */}
                      <Grid item xs={12} md={6}>
                        <Typography variant="subtitle1" fontWeight="bold" gutterBottom color="error.main">
                          ➖ Top Detracting Features
                        </Typography>
                        {prediction.explanation.top_negative_features
                          .slice(0, 5)
                          .map((feature, index) => (
                            <Box key={feature.feature_name} sx={{ mb: 1.5 }}>
                              <Box display="flex" justifyContent="space-between" alignItems="center" mb={0.5}>
                                <Typography variant="body2" fontWeight="medium">
                                  {feature.feature_name}
                                </Typography>
                                <Box display="flex" alignItems="center" gap={1}>
                                  <Chip
                                    label={`#${feature.importance_rank}`}
                                    size="small"
                                    variant="outlined"
                                    color="primary"
                                  />
                                  <Typography variant="caption" fontWeight="bold" color="error.main">
                                    {feature.shap_value.toFixed(3)}
                                  </Typography>
                                </Box>
                              </Box>
                              <Box display="flex" alignItems="center" gap={1}>
                                <LinearProgress
                                  variant="determinate"
                                  value={Math.min(Math.abs(feature.shap_value) * 1000, 100)}
                                  sx={{ 
                                    flexGrow: 1, 
                                    height: 6, 
                                    borderRadius: 3,
                                    '& .MuiLinearProgress-bar': {
                                      backgroundColor: '#f44336'
                                    }
                                  }}
                                />
                                <Typography variant="caption" color="text.secondary">
                                  Value: {feature.feature_value.toFixed(2)}
                                </Typography>
                              </Box>
                            </Box>
                          ))}
                      </Grid>
                    </Grid>
                    
                    <Alert severity="info" sx={{ mt: 2 }}>
                      <Typography variant="body2">
                        <strong>SHAP Explanation:</strong> Positive values increase the likelihood of the predicted cancer type, 
                        while negative values decrease it. The larger the absolute value, the more important the feature is for this prediction.
                      </Typography>
                    </Alert>
                  </>
                )}
                
                <Divider sx={{ my: 2 }} />
                
                <Box display="flex" justifyContent="space-between" alignItems="center">
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Model: {prediction.model_used}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Processing Time: {prediction.processing_time_ms.toFixed(2)}ms
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    {new Date(prediction.timestamp).toLocaleString()}
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default PredictionInterface;
