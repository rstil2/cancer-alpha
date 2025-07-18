import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Chip,
  CircularProgress,
  Alert,
  LinearProgress,
} from '@mui/material';
import {
  CheckCircle,
  Error,
  Warning,
  Analytics,
  ModelTraining,
  Security,
} from '@mui/icons-material';
import { apiService } from '../services/api';
import { HealthStatus, ModelInfo, CancerTypesResponse } from '../types/api';

const Dashboard: React.FC = () => {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [cancerTypes, setCancerTypes] = useState<CancerTypesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Fetch all dashboard data
        const [healthData, modelData, cancerTypesData] = await Promise.all([
          apiService.checkHealth(),
          apiService.getModelInfo(),
          apiService.getCancerTypes(),
        ]);

        setHealth(healthData);
        setModelInfo(modelData);
        setCancerTypes(cancerTypesData);
      } catch (err: any) {
        setError(err.detail || 'Failed to fetch dashboard data');
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'healthy':
        return 'success';
      case 'unhealthy':
        return 'error';
      default:
        return 'warning';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case 'healthy':
        return <CheckCircle color="success" />;
      case 'unhealthy':
        return <Error color="error" />;
      default:
        return <Warning color="warning" />;
    }
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
        Cancer Alpha Dashboard
      </Typography>

      <Grid container spacing={3}>
        {/* System Health Card */}
        <Grid item xs={12} md={6} lg={4}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                {health && getStatusIcon(health.status)}
                <Typography variant="h6" component="h2" sx={{ ml: 1 }}>
                  System Health
                </Typography>
              </Box>
              {health && (
                <>
                  <Chip
                    label={health.status.toUpperCase()}
                    color={getStatusColor(health.status) as any}
                    sx={{ mb: 2 }}
                  />
                  <Typography variant="body2" color="text.secondary">
                    {health.message}
                  </Typography>
                  <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                    Last updated: {new Date(health.timestamp).toLocaleString()}
                  </Typography>
                </>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Model Status Card */}
        <Grid item xs={12} md={6} lg={4}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <ModelTraining color="primary" />
                <Typography variant="h6" component="h2" sx={{ ml: 1 }}>
                  Model Status
                </Typography>
              </Box>
              {modelInfo && (
                <>
                  <Typography variant="body2" gutterBottom>
                    Loaded Models: {modelInfo.loaded_models.length}
                  </Typography>
                  <Box sx={{ mb: 2 }}>
                    {modelInfo.loaded_models.map((model) => (
                      <Chip
                        key={model}
                        label={model.replace('_', ' ').toUpperCase()}
                        size="small"
                        sx={{ mr: 1, mb: 1 }}
                      />
                    ))}
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    Features: {modelInfo.feature_count || 'N/A'}
                  </Typography>
                </>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Cancer Types Card */}
        <Grid item xs={12} md={6} lg={4}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Analytics color="secondary" />
                <Typography variant="h6" component="h2" sx={{ ml: 1 }}>
                  Cancer Types
                </Typography>
              </Box>
              {cancerTypes && (
                <>
                  <Typography variant="h4" component="p" gutterBottom>
                    {cancerTypes.total_types}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Supported cancer types for classification
                  </Typography>
                  <Box sx={{ mt: 2 }}>
                    {cancerTypes.cancer_types.slice(0, 4).map((type) => (
                      <Chip
                        key={type}
                        label={type}
                        size="small"
                        variant="outlined"
                        sx={{ mr: 1, mb: 1 }}
                      />
                    ))}
                    {cancerTypes.cancer_types.length > 4 && (
                      <Chip
                        label={`+${cancerTypes.cancer_types.length - 4} more`}
                        size="small"
                        variant="outlined"
                        sx={{ mr: 1, mb: 1 }}
                      />
                    )}
                  </Box>
                </>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Model Performance Card */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                Model Performance
              </Typography>
              {modelInfo?.model_performance && (
                <Grid container spacing={2}>
                  {Object.entries(modelInfo.model_performance).map(([modelName, performance]) => (
                    <Grid item xs={12} sm={6} md={3} key={modelName}>
                      <Box>
                        <Typography variant="body2" fontWeight="bold">
                          {modelName.replace('_', ' ').toUpperCase()}
                        </Typography>
                        <Typography variant="h6" color="primary">
                          {(performance.test_accuracy * 100).toFixed(1)}%
                        </Typography>
                        <LinearProgress
                          variant="determinate"
                          value={performance.test_accuracy * 100}
                          sx={{ mt: 1 }}
                        />
                      </Box>
                    </Grid>
                  ))}
                </Grid>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* System Information Card */}
        <Grid item xs={12} lg={4}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Security color="primary" />
                <Typography variant="h6" component="h2" sx={{ ml: 1 }}>
                  System Information
                </Typography>
              </Box>
              <Typography variant="body2" gutterBottom>
                API Version: 2.0.0
              </Typography>
              <Typography variant="body2" gutterBottom>
                Framework: Cancer Alpha
              </Typography>
              <Typography variant="body2" gutterBottom>
                Environment: {process.env.NODE_ENV || 'development'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Real-time cancer classification using multi-modal genomic data
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
