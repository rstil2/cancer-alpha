import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { AppBar, Toolbar, Typography, Container } from '@mui/material';

// Components
import Dashboard from './pages/Dashboard';
import PredictionInterface from './pages/PredictionInterface';
import ModelManagement from './pages/ModelManagement';
import ResultsView from './pages/ResultsView';
import Navigation from './components/Navigation';

// Theme configuration
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    h4: {
      fontWeight: 600,
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <AppBar position="static" elevation={1}>
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              Cancer Alpha - Multi-Modal Cancer Classification
            </Typography>
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
