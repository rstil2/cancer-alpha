import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { AppBar, Toolbar, Typography, Container, Box, Chip } from '@mui/material';
import { Science, Biotech } from '@mui/icons-material';

// Components
import Dashboard from './pages/Dashboard';
import PredictionInterface from './pages/PredictionInterface';
import ModelManagement from './pages/ModelManagement';
import ResultsView from './pages/ResultsView';
import Navigation from './components/Navigation';

// Enhanced Theme configuration for Cancer Alpha
const theme = createTheme({
  palette: {
    primary: {
      main: '#2196F3', // Cancer Alpha blue
      light: '#64B5F6',
      dark: '#1976D2',
    },
    secondary: {
      main: '#E91E63', // Cancer Alpha pink  
      light: '#F48FB1',
      dark: '#C2185B',
    },
    success: {
      main: '#4CAF50',
      light: '#81C784',
      dark: '#388E3C',
    },
    warning: {
      main: '#FF9800',
      light: '#FFB74D',
      dark: '#F57C00',
    },
    error: {
      main: '#F44336',
      light: '#EF5350',
      dark: '#D32F2F',
    },
    background: {
      default: '#F8F9FA',
      paper: '#FFFFFF',
    },
    text: {
      primary: '#1A1A1A',
      secondary: '#6C757D',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 700,
      fontSize: '2.5rem',
    },
    h2: {
      fontWeight: 600,
      fontSize: '2rem',
    },
    h3: {
      fontWeight: 600,
      fontSize: '1.75rem',
    },
    h4: {
      fontWeight: 600,
      fontSize: '1.5rem',
    },
    h5: {
      fontWeight: 600,
      fontSize: '1.25rem',
    },
    h6: {
      fontWeight: 600,
      fontSize: '1.125rem',
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.5,
    },
  },
  shape: {
    borderRadius: 12,
  },
  shadows: [
    'none',
    '0px 2px 4px rgba(0,0,0,0.05)',
    '0px 4px 8px rgba(0,0,0,0.1)',
    '0px 8px 16px rgba(0,0,0,0.15)',
    '0px 16px 32px rgba(0,0,0,0.2)',
    // ... other shadow definitions can be added
  ] as any,
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: 8,
          boxShadow: 'none',
          '&:hover': {
            boxShadow: '0px 4px 8px rgba(0,0,0,0.1)',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0px 2px 8px rgba(0,0,0,0.06)',
          border: '1px solid #E5E7EB',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: '#FFFFFF',
          color: '#1A1A1A',
          boxShadow: '0px 1px 3px rgba(0,0,0,0.1)',
        },
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <AppBar position="static" elevation={0}>
          <Toolbar sx={{ py: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexGrow: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Biotech sx={{ fontSize: 32, color: theme.palette.primary.main }} />
                <Box>
                  <Typography variant="h5" component="div" sx={{ fontWeight: 700, lineHeight: 1 }}>
                    Cancer Alpha
                  </Typography>
                  <Typography variant="body2" sx={{ color: 'text.secondary', lineHeight: 1 }}>
                    AI-Powered Cancer Classification
                  </Typography>
                </Box>
              </Box>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Chip 
                icon={<Science />}
                label="Research Use Only" 
                color="warning" 
                variant="outlined" 
                size="small"
              />
              <Chip 
                label="v2.0 - Real Models" 
                color="success" 
                size="small"
              />
            </Box>
          </Toolbar>
        </AppBar>
        
        <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
          <Navigation />
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/predict" element={<PredictionInterface />} />
            <Route path="/models" element={<ModelManagement />} />
            <Route path="/results" element={<ResultsView />} />
          </Routes>
        </Container>
      </Router>
    </ThemeProvider>
  );
}

export default App;
