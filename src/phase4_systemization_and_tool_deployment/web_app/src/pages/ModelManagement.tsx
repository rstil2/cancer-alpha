import React, { useState, useEffect } from 'react';
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
  CircularProgress,
  Alert,
  LinearProgress,
  Divider,
} from '@mui/material';
import {
  ModelTraining,
  Analytics,
  TrendingUp,
  Speed,
} from '@mui/icons-material';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { apiService } from '../services/api';
import { ModelInfo, FeatureImportance } from '../types/api';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const ModelManagement: React.FC = () => {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchModelData = async () => {
      try {
        setLoading(true);
        setError(null);

        const [modelData, featureData] = await Promise.all([
          apiService.getModelInfo(),
          apiService.getFeatureImportance().catch(() => null), // Feature importance might not be available
        ]);

        setModelInfo(modelData);
        setFeatureImportance(featureData);
      } catch (err: any) {
        setError(err.detail || 'Failed to fetch model information');
      } finally {
        setLoading(false);
      }
    };

    fetchModelData();
  }, []);

  const getFeatureImportanceChartData = () => {
    if (!featureImportance) return null;

    const features = Object.entries(featureImportance.top_features)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 10);

    return {
      labels: features.map(([name]) => name.replace('_', ' ')),
      datasets: [
        {
          label: 'Feature Importance',
          data: features.map(([, importance]) => importance),
          backgroundColor: 'rgba(25, 118, 210, 0.8)',
          borderColor: 'rgba(25, 118, 210, 1)',
          borderWidth: 1,
        },
      ],
    };
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Top 10 Most Important Features',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Model Management
      </Typography>

      <Grid container spacing={3}>
        {/* Model Status Overview */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <ModelTraining color="primary" />
                <Typography variant="h6" component="h2" sx={{ ml: 1 }}>
                  Model Status
                </Typography>
              </Box>
              {modelInfo && (
                <Box>
                  <Typography variant="body2" gutterBottom>
                    Loaded Models: {modelInfo.loaded_models.length}
                  </Typography>
                  <Typography variant="body2" gutterBottom>
                    Scaler: {modelInfo.scaler_loaded ? 'Loaded' : 'Not Loaded'}
                  </Typography>
                  <Typography variant="body2" gutterBottom>
                    Feature Importance: {modelInfo.feature_importance_loaded ? 'Available' : 'Not Available'}
                  </Typography>
                  <Typography variant="body2" gutterBottom>
                    Metadata: {modelInfo.metadata_loaded ? 'Available' : 'Not Available'}
                  </Typography>
                  <Typography variant="body2">
                    Features: {modelInfo.feature_count || 'N/A'}
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Feature Summary */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Analytics color="secondary" />
                <Typography variant="h6" component="h2" sx={{ ml: 1 }}>
                  Feature Summary
                </Typography>
              </Box>
              {featureImportance && (
                <Box>
                  <Typography variant="h4" component="p" gutterBottom>
                    {featureImportance.total_features}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Total Features Analyzed
                  </Typography>
                  <Typography variant="body2" gutterBottom>
                    Scale: {featureImportance.importance_scale}
                  </Typography>
                  <Typography variant="body2">
                    Top Features: {Object.keys(featureImportance.top_features).length}
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Performance Summary */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <TrendingUp color="success" />
                <Typography variant="h6" component="h2" sx={{ ml: 1 }}>
                  Performance Summary
                </Typography>
              </Box>
              {modelInfo?.model_performance && (
                <Box>
                  <Typography variant="body2" gutterBottom>
                    Best Model: {
                      Object.entries(modelInfo.model_performance)
                        .sort(([, a], [, b]) => b.test_accuracy - a.test_accuracy)[0]?.[0]
                        ?.replace('_', ' ').toUpperCase()
                    }
                  </Typography>
                  <Typography variant="body2" gutterBottom>
                    Best Accuracy: {
                      Math.max(...Object.values(modelInfo.model_performance)
                        .map(p => p.test_accuracy)) * 100
                    }%
                  </Typography>
                  <Typography variant="body2">
                    Total Models: {Object.keys(modelInfo.model_performance).length}
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Model Performance Table */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                Model Performance Details
              </Typography>
              {modelInfo?.model_performance && (
                <TableContainer component={Paper} elevation={0}>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Model</TableCell>
                        <TableCell align="right">Test Accuracy</TableCell>
                        <TableCell align="right">CV Mean</TableCell>
                        <TableCell align="right">Performance</TableCell>
                        <TableCell align="right">Status</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {Object.entries(modelInfo.model_performance)
                        .sort(([, a], [, b]) => b.test_accuracy - a.test_accuracy)
                        .map(([modelName, performance]) => (
                          <TableRow key={modelName}>
                            <TableCell component="th" scope="row">
                              <Typography variant="body2" fontWeight="bold">
                                {modelName.replace('_', ' ').toUpperCase()}
                              </Typography>
                            </TableCell>
                            <TableCell align="right">
                              <Typography variant="body2">
                                {(performance.test_accuracy * 100).toFixed(1)}%
                              </Typography>
                            </TableCell>
                            <TableCell align="right">
                              <Typography variant="body2">
                                {performance.cv_mean 
                                  ? (performance.cv_mean * 100).toFixed(1) + '%'
                                  : 'N/A'}
                              </Typography>
                            </TableCell>
                            <TableCell align="right">
                              <Box sx={{ width: 100 }}>
                                <LinearProgress
                                  variant="determinate"
                                  value={performance.test_accuracy * 100}
                                  sx={{ height: 8, borderRadius: 4 }}
                                />
                              </Box>
                            </TableCell>
                            <TableCell align="right">
                              <Chip
                                label={performance.test_accuracy > 0.8 ? 'Excellent' : 
                                      performance.test_accuracy > 0.6 ? 'Good' : 'Fair'}
                                color={performance.test_accuracy > 0.8 ? 'success' : 
                                      performance.test_accuracy > 0.6 ? 'primary' : 'warning'}
                                size="small"
                              />
                            </TableCell>
                          </TableRow>
                        ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Model List */}
        <Grid item xs={12} lg={4}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Speed color="info" />
                <Typography variant="h6" component="h2" sx={{ ml: 1 }}>
                  Available Models
                </Typography>
              </Box>
              {modelInfo && (
                <Box>
                  {modelInfo.loaded_models.map((model, index) => (
                    <Box key={model} mb={1}>
                      <Chip
                        label={model.replace('_', ' ').toUpperCase()}
                        color="primary"
                        variant="outlined"
                        sx={{ width: '100%', justifyContent: 'flex-start' }}
                      />
                    </Box>
                  ))}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Feature Importance Chart */}
        {featureImportance && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" component="h2" gutterBottom>
                  Feature Importance Analysis
                </Typography>
                <Divider sx={{ mb: 2 }} />
                <Box sx={{ height: 400 }}>
                  <Bar data={getFeatureImportanceChartData()!} options={chartOptions} />
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Top Features List */}
        {featureImportance && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" component="h2" gutterBottom>
                  Top Features Detail
                </Typography>
                <Grid container spacing={2}>
                  {Object.entries(featureImportance.top_features)
                    .sort(([, a], [, b]) => b - a)
                    .map(([feature, importance], index) => (
                      <Grid item xs={12} sm={6} md={4} lg={3} key={feature}>
                        <Box p={2} border={1} borderColor="grey.300" borderRadius={2}>
                          <Typography variant="body2" fontWeight="bold" gutterBottom>
                            {feature.replace('_', ' ').toUpperCase()}
                          </Typography>
                          <Typography variant="h6" color="primary">
                            {importance.toFixed(4)}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Rank #{index + 1}
                          </Typography>
                        </Box>
                      </Grid>
                    ))}
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default ModelManagement;
