import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { 
  Box, 
  Tabs, 
  Tab, 
  Paper 
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Psychology as PredictIcon,
  ModelTraining as ModelIcon,
  Assessment as ResultsIcon,
} from '@mui/icons-material';

const Navigation: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const getCurrentTab = () => {
    switch (location.pathname) {
      case '/':
        return 0;
      case '/predict':
        return 1;
      case '/models':
        return 2;
      case '/results':
        return 3;
      default:
        return 0;
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    const routes = ['/', '/predict', '/models', '/results'];
    navigate(routes[newValue]);
  };

  return (
    <Paper elevation={1} sx={{ mb: 3 }}>
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs 
          value={getCurrentTab()} 
          onChange={handleTabChange}
          aria-label="navigation tabs"
        >
          <Tab 
            icon={<DashboardIcon />} 
            label="Dashboard" 
            iconPosition="start"
          />
          <Tab 
            icon={<PredictIcon />} 
            label="Prediction" 
            iconPosition="start"
          />
          <Tab 
            icon={<ModelIcon />} 
            label="Models" 
            iconPosition="start"
          />
          <Tab 
            icon={<ResultsIcon />} 
            label="Results" 
            iconPosition="start"
          />
        </Tabs>
      </Box>
    </Paper>
  );
};

export default Navigation;
