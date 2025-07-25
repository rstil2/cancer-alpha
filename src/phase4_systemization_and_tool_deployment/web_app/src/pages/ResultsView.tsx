import React, { useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Button,
  Alert,
  Divider,
  LinearProgress,
} from '@mui/material';
import {
  History,
  Analytics,
  Download,
  Visibility,
  TrendingUp,
} from '@mui/icons-material';
import { PredictionResponse } from '../types/api';

// Mock data for demonstration
const mockPredictions: PredictionResponse[] = [
  {
    patient_id: 'P001',
    predicted_cancer_type: 'BRCA',
    predicted_cancer_name: 'Breast Invasive Carcinoma',
    confidence_metrics: {
      prediction_confidence: 0.95,
      confidence_level: 'High',
      entropy: 0.2,
      top_2_margin: 0.89
    },
    probability_distribution: {
      'BRCA': 0.95,
      'LUAD': 0.02,
      'COAD': 0.01,
      'PRAD': 0.01,
      'STAD': 0.01,
      'KIRC': 0.0,
      'HNSC': 0.0,
      'LIHC': 0.0,
    },
    explanation: {
      top_positive_features: [],
      top_negative_features: [],
      explanation_available: false,
      explanation_method: 'Not available',
      base_value: 0.0,
      prediction_value: 0.0
    },
    model_used: 'ensemble',
    timestamp: '2025-07-18T10:30:00Z',
    processing_time_ms: 45.2,
    model_accuracy: 0.99,
    confidence: 0.95, // Legacy compatibility
  },
  {
    patient_id: 'P002',
    predicted_cancer_type: 'LUAD',
    predicted_cancer_name: 'Lung Adenocarcinoma',
    confidence_metrics: {
      prediction_confidence: 0.87,
      confidence_level: 'High',
      entropy: 0.35,
      top_2_margin: 0.76
    },
    probability_distribution: {
      'LUAD': 0.87,
      'BRCA': 0.05,
      'COAD': 0.03,
      'PRAD': 0.02,
      'STAD': 0.02,
      'KIRC': 0.01,
      'HNSC': 0.0,
      'LIHC': 0.0,
    },
    explanation: {
      top_positive_features: [],
      top_negative_features: [],
      explanation_available: false,
      explanation_method: 'Not available',
      base_value: 0.0,
      prediction_value: 0.0
    },
    model_used: 'random_forest',
    timestamp: '2025-07-18T11:15:00Z',
    processing_time_ms: 38.7,
    model_accuracy: 0.92,
    confidence: 0.87, // Legacy compatibility
  },
  {
    patient_id: 'P003',
    predicted_cancer_type: 'COAD',
    predicted_cancer_name: 'Colon Adenocarcinoma',
    confidence_metrics: {
      prediction_confidence: 0.92,
      confidence_level: 'Very High',
      entropy: 0.25,
      top_2_margin: 0.84
    },
    probability_distribution: {
      'COAD': 0.92,
      'STAD': 0.04,
      'BRCA': 0.02,
      'LUAD': 0.01,
      'PRAD': 0.01,
      'KIRC': 0.0,
      'HNSC': 0.0,
      'LIHC': 0.0,
    },
    explanation: {
      top_positive_features: [],
      top_negative_features: [],
      explanation_available: false,
      explanation_method: 'Not available',
      base_value: 0.0,
      prediction_value: 0.0
    },
    model_used: 'gradient_boosting',
    timestamp: '2025-07-18T12:00:00Z',
    processing_time_ms: 42.1,
    model_accuracy: 0.96,
    confidence: 0.92, // Legacy compatibility
  },
];

const ResultsView: React.FC = () => {
  const [predictions] = useState<PredictionResponse[]>(mockPredictions);
  const [selectedPrediction, setSelectedPrediction] = useState<PredictionResponse | null>(null);

  // Helper function for backward compatibility with confidence field
  const getConfidence = (prediction: PredictionResponse): number => {
    return prediction.confidence_metrics?.prediction_confidence ?? prediction.confidence ?? 0;
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return 'success';
    if (confidence >= 0.7) return 'primary';
    if (confidence >= 0.5) return 'warning';
    return 'error';
  };

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.9) return 'Very High';
    if (confidence >= 0.7) return 'High';
    if (confidence >= 0.5) return 'Medium';
    return 'Low';
  };

  const exportResults = () => {
    const csvContent = [
      ['Patient ID', 'Predicted Type', 'Confidence', 'Model Used', 'Timestamp'].join(','),
      ...predictions.map(p => [
        p.patient_id,
        p.predicted_cancer_type,
        getConfidence(p).toFixed(3),
        p.model_used,
        new Date(p.timestamp).toISOString()
      ].join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'cancer_predictions.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const cancerTypeDistribution = predictions.reduce((acc, pred) => {
    acc[pred.predicted_cancer_type] = (acc[pred.predicted_cancer_type] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const averageConfidence = predictions.reduce((sum, pred) => sum + getConfidence(pred), 0) / predictions.length;
  const averageProcessingTime = predictions.reduce((sum, pred) => sum + pred.processing_time_ms, 0) / predictions.length;

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Results & Analytics
      </Typography>

      <Grid container spacing={3}>
        {/* Summary Statistics */}
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <History color="primary" />
                <Typography variant="h6" component="h2" sx={{ ml: 1 }}>
                  Total Predictions
                </Typography>
              </Box>
              <Typography variant="h3" component="p" color="primary">
                {predictions.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <TrendingUp color="success" />
                <Typography variant="h6" component="h2" sx={{ ml: 1 }}>
                  Avg. Confidence
                </Typography>
              </Box>
              <Typography variant="h3" component="p" color="success.main">
                {(averageConfidence * 100).toFixed(1)}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Analytics color="secondary" />
                <Typography variant="h6" component="h2" sx={{ ml: 1 }}>
                  Avg. Processing
                </Typography>
              </Box>
              <Typography variant="h3" component="p" color="secondary.main">
                {averageProcessingTime.toFixed(0)}ms
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Download color="info" />
                <Typography variant="h6" component="h2" sx={{ ml: 1 }}>
                  Export Data
                </Typography>
              </Box>
              <Button
                variant="contained"
                startIcon={<Download />}
                onClick={exportResults}
                size="small"
              >
                Export CSV
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {/* Cancer Type Distribution */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                Cancer Type Distribution
              </Typography>
              <Box>
                {Object.entries(cancerTypeDistribution).map(([type, count]) => (
                  <Box key={type} sx={{ mb: 2 }}>
                    <Box display="flex" justifyContent="space-between" mb={1}>
                      <Typography variant="body2">{type}</Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {count} ({((count / predictions.length) * 100).toFixed(1)}%)
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={(count / predictions.length) * 100}
                      sx={{ height: 8, borderRadius: 4 }}
                    />
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Activity */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                Recent Activity
              </Typography>
              <Alert severity="info" sx={{ mb: 2 }}>
                This is a demonstration with mock data. In production, this would show real prediction history.
              </Alert>
              <Box>
                {predictions.slice(0, 3).map((prediction, index) => (
                  <Box key={prediction.patient_id} sx={{ mb: 2 }}>
                    <Box display="flex" justifyContent="space-between" alignItems="center">
                      <Box>
                        <Typography variant="body2" fontWeight="bold">
                          Patient {prediction.patient_id}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {new Date(prediction.timestamp).toLocaleString()}
                        </Typography>
                      </Box>
                      <Chip
                        label={prediction.predicted_cancer_type}
                        color={getConfidenceColor(getConfidence(prediction)) as any}
                        size="small"
                      />
                    </Box>
                    {index < 2 && <Divider sx={{ mt: 1 }} />}
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Predictions Table */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                <Typography variant="h6" component="h2">
                  Prediction History
                </Typography>
                <Button
                  variant="outlined"
                  startIcon={<Download />}
                  onClick={exportResults}
                  size="small"
                >
                  Export All
                </Button>
              </Box>
              <TableContainer component={Paper} elevation={0}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Patient ID</TableCell>
                      <TableCell>Predicted Type</TableCell>
                      <TableCell>Predicted Name</TableCell>
                      <TableCell align="right">Confidence</TableCell>
                      <TableCell align="right">Model Used</TableCell>
                      <TableCell align="right">Processing Time</TableCell>
                      <TableCell align="right">Timestamp</TableCell>
                      <TableCell align="right">Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {predictions.map((prediction) => (
                      <TableRow
                        key={prediction.patient_id}
                        hover
                        sx={{ cursor: 'pointer' }}
                        onClick={() => setSelectedPrediction(prediction)}
                      >
                        <TableCell component="th" scope="row">
                          <Typography variant="body2" fontWeight="bold">
                            {prediction.patient_id}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={prediction.predicted_cancer_type}
                            color={getConfidenceColor(getConfidence(prediction)) as any}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2">
                            {prediction.predicted_cancer_name}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Box display="flex" alignItems="center" justifyContent="flex-end">
                            <Typography variant="body2" sx={{ mr: 1 }}>
                              {(getConfidence(prediction) * 100).toFixed(1)}%
                            </Typography>
                            <Chip
                              label={getConfidenceLabel(getConfidence(prediction))}
                              color={getConfidenceColor(getConfidence(prediction)) as any}
                              size="small"
                            />
                          </Box>
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="body2">
                            {prediction.model_used}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="body2">
                            {prediction.processing_time_ms.toFixed(1)}ms
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="body2">
                            {new Date(prediction.timestamp).toLocaleString()}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Button
                            size="small"
                            startIcon={<Visibility />}
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedPrediction(prediction);
                            }}
                          >
                            View
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Selected Prediction Details */}
        {selectedPrediction && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                  <Typography variant="h6" component="h2">
                    Prediction Details - {selectedPrediction.patient_id}
                  </Typography>
                  <Button
                    variant="outlined"
                    onClick={() => setSelectedPrediction(null)}
                    size="small"
                  >
                    Close
                  </Button>
                </Box>
                <Divider sx={{ mb: 2 }} />
                
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Box>
                      <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                        Prediction Summary
                      </Typography>
                      <Typography variant="body2" gutterBottom>
                        <strong>Cancer Type:</strong> {selectedPrediction.predicted_cancer_type}
                      </Typography>
                      <Typography variant="body2" gutterBottom>
                        <strong>Full Name:</strong> {selectedPrediction.predicted_cancer_name}
                      </Typography>
                      <Typography variant="body2" gutterBottom>
                        <strong>Confidence:</strong> {(getConfidence(selectedPrediction) * 100).toFixed(1)}%
                      </Typography>
                      <Typography variant="body2" gutterBottom>
                        <strong>Model Used:</strong> {selectedPrediction.model_used}
                      </Typography>
                      <Typography variant="body2" gutterBottom>
                        <strong>Processing Time:</strong> {selectedPrediction.processing_time_ms.toFixed(2)}ms
                      </Typography>
                      <Typography variant="body2">
                        <strong>Timestamp:</strong> {new Date(selectedPrediction.timestamp).toLocaleString()}
                      </Typography>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                      Probability Distribution
                    </Typography>
                    <Box>
                      {Object.entries(selectedPrediction.probability_distribution)
                        .sort(([, a], [, b]) => b - a)
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
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default ResultsView;
