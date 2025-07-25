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
  TextField,
  MenuItem,
  CircularProgress,
  Alert,
  InputAdornment,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  LinearProgress,
} from '@mui/material';
import {
  Search,
  ExpandMore,
  Info,
  Biotech,
  Analytics,
  DataUsage,
  Verified,
} from '@mui/icons-material';
import { apiService } from '../services/api';
import { FeatureInfoResponse, FeatureCategory } from '../types/api';

const Features: React.FC = () => {
  const [featureData, setFeatureData] = useState<FeatureInfoResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedCategory, setSelectedCategory] = useState('All');

  useEffect(() => {
    const fetchFeatureData = async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await apiService.getFeatureInfo();
        setFeatureData(data);
      } catch (err: any) {
        setError(err.detail || 'Failed to fetch feature information');
      } finally {
        setLoading(false);
      }
    };

    fetchFeatureData();
  }, []);

  const getCategoryColor = (category: string) => {
    const colors: Record<string, string> = {
      'methylation': 'primary',
      'mutation': 'secondary',
      'copynumber': 'info',
      'fragment': 'success',
      'clinical': 'warning',
      'icgc': 'error',
    };
    return colors[category.toLowerCase()] || 'default';
  };

  const getCategoryDisplayName = (category: string) => {
    const names: Record<string, string> = {
      'methylation': 'Methylation',
      'mutation': 'Mutation',
      'copynumber': 'Copy Number',
      'fragment': 'Fragment',
      'clinical': 'Clinical',
      'icgc': 'ICGC Pathway',
    };
    return names[category.toLowerCase()] || category;
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

  if (!featureData) {
    return (
      <Alert severity="warning" sx={{ mb: 2 }}>
        No feature data available
      </Alert>
    );
  }

  const categories = Object.keys(featureData.feature_categories);
  const filteredCategories = selectedCategory === 'All' 
    ? categories 
    : categories.filter(cat => cat === selectedCategory);

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Genomic Features Information
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Comprehensive overview of the {featureData.total_features} genomic features used for cancer classification
      </Typography>

      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* Summary Statistics */}
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <DataUsage color="primary" />
                <Typography variant="h6" component="h2" sx={{ ml: 1 }}>
                  Total Features
                </Typography>
              </Box>
              <Typography variant="h3" component="p" color="primary">
                {featureData.total_features}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Verified color="success" />
                <Typography variant="h6" component="h2" sx={{ ml: 1 }}>
                  SHAP Support
                </Typography>
              </Box>
              <Typography variant="h3" component="p" color="success.main">
                {featureData.explainability.shap_support ? 'Yes' : 'No'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Explainable AI enabled
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Biotech color="secondary" />
                <Typography variant="h6" component="h2" sx={{ ml: 1 }}>
                  Categories
                </Typography>
              </Box>
              <Typography variant="h3" component="p" color="secondary.main">
                {categories.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Analytics color="info" />
                <Typography variant="h6" component="h2" sx={{ ml: 1 }}>
                  Models
                </Typography>
              </Box>
              <Typography variant="body1" component="p" sx={{ fontWeight: 600 }}>
                {featureData.explainability.available_explainers.length} Explainers
              </Typography>
              <Typography variant="body2" color="text.secondary">
                AI interpretability
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Feature Categories */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" component="h2" gutterBottom>
            Filter by Category
          </Typography>
          <TextField
            fullWidth
            select
            label="Select Category"
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            sx={{ mb: 2 }}
          >
            <MenuItem value="All">All Categories</MenuItem>
            {categories.map((category) => (
              <MenuItem key={category} value={category}>
                {getCategoryDisplayName(category)}
              </MenuItem>
            ))}
          </TextField>
          
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {categories.map((category) => {
              const categoryData = featureData.feature_categories[category];
              return (
                <Chip
                  key={category}
                  label={`${getCategoryDisplayName(category)} (${categoryData.count})`}
                  color={getCategoryColor(category) as any}
                  variant={selectedCategory === category ? 'filled' : 'outlined'}
                  onClick={() => setSelectedCategory(selectedCategory === category ? 'All' : category)}
                  clickable
                />
              );
            })}
          </Box>
        </CardContent>
      </Card>

      {/* Category Details */}
      {filteredCategories.map((categoryKey) => {
        const category = featureData.feature_categories[categoryKey];
        return (
          <Card key={categoryKey} sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                {getCategoryDisplayName(categoryKey)} Features
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                {category.description}
              </Typography>
              
              <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid item xs={12} md={3}>
                  <Typography variant="body2" color="text.secondary">Count</Typography>
                  <Typography variant="h6">{category.count}</Typography>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Typography variant="body2" color="text.secondary">Range</Typography>
                  <Typography variant="h6">{category.range}</Typography>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="body2" color="text.secondary">Clinical Relevance</Typography>
                  <Typography variant="body2">{category.clinical_relevance}</Typography>
                </Grid>
              </Grid>
              
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>Example Features:</Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                {category.example_features.map((feature) => (
                  <Chip
                    key={feature}
                    label={feature}
                    size="small"
                    variant="outlined"
                  />
                ))}
              </Box>
            </CardContent>
          </Card>
        );
      })}

      {/* Explainability Information */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" component="h2" gutterBottom>
            Explainability Information
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="body2" color="text.secondary">SHAP Support</Typography>
              <Typography variant="body1">
                {featureData.explainability.shap_support ? '✅ Enabled' : '❌ Disabled'}
              </Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="body2" color="text.secondary">Available Explainers</Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                {featureData.explainability.available_explainers.map((explainer) => (
                  <Chip
                    key={explainer}
                    label={explainer}
                    size="small"
                    color="primary"
                    variant="outlined"
                  />
                ))}
              </Box>
            </Grid>
          </Grid>
          
          <Typography variant="body2" color="text.secondary" sx={{ mt: 2, mb: 1 }}>Explanation Methods:</Typography>
          {Object.entries(featureData.explainability.explanation_methods).map(([method, description]) => (
            <Box key={method} sx={{ mb: 1 }}>
              <Typography variant="body2" fontWeight="bold">{method}:</Typography>
              <Typography variant="body2" color="text.secondary">{description}</Typography>
            </Box>
          ))}
        </CardContent>
      </Card>

      {/* Usage Notes */}
      <Card>
        <CardContent>
          <Typography variant="h6" component="h2" gutterBottom>
            Usage Notes
          </Typography>
          <Box component="ul" sx={{ pl: 2 }}>
            {featureData.usage_notes.map((note, index) => (
              <Box component="li" key={index} sx={{ mb: 1 }}>
                <Typography variant="body2">{note}</Typography>
              </Box>
            ))}
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Features;
