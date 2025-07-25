# Cancer Alpha Web App - Features Tab Guide

**Version 2.0.0 - Updated for SHAP Explainability**  
**Last Updated:** July 25, 2025

---

## 🧬 **Overview**

The **Features** tab in the Cancer Alpha web application provides comprehensive insights into the 110 genomic features used by the machine learning models for cancer classification. This tab serves as both an educational resource and a technical reference for researchers and clinicians.

### Key Capabilities
- **Feature Category Breakdown**: Detailed exploration of all 6 genomic feature categories
- **SHAP Explainability Status**: View which features support AI interpretability
- **Clinical Relevance Information**: Understand the biological significance of each feature category
- **Interactive Filtering**: Explore features by category with dynamic filtering
- **Usage Guidelines**: Best practices for feature data preparation

---

## 🗂️ **Feature Categories**

The Cancer Alpha system analyzes **110 genomic features** organized into 6 major categories:

### 1. **🧬 Methylation Features (20 features)**
- **Range**: Features 0-19 (`methylation_0` to `methylation_19`)
- **Data Type**: Float values (0.0 - 1.0)
- **Description**: DNA methylation patterns representing methylation levels
- **Clinical Relevance**: Hypermethylation often associated with tumor suppressor gene silencing

### 2. **🧬 Mutation Features (25 features)**
- **Range**: Features 20-44 (`mutation_0` to `mutation_24`)
- **Data Type**: Integer counts (0 - 50+)
- **Description**: Genetic mutation counts representing mutation burden
- **Clinical Relevance**: High mutation burden may indicate genomic instability

### 3. **🧬 Copy Number Features (20 features)**
- **Range**: Features 45-64 (`copynumber_0` to `copynumber_19`)
- **Data Type**: Float values (0.0 - 10.0)
- **Description**: Copy number variations - typically around 2 (diploid) with amplifications/deletions
- **Clinical Relevance**: Copy number alterations drive oncogene activation or tumor suppressor loss

### 4. **🧬 Fragment Features (15 features)**
- **Range**: Features 65-79 (`fragment_0` to `fragment_14`)
- **Data Type**: Float values (50.0 - 500.0)
- **Description**: Circulating tumor DNA fragmentomics from liquid biopsy analysis
- **Clinical Relevance**: Fragment patterns reflect tumor biology and treatment response

### 5. **🧬 Clinical Features (10 features)**
- **Range**: Features 80-89 (`clinical_0` to `clinical_9`)
- **Data Type**: Float values (0.0 - 1.0)
- **Description**: Clinical variables - normalized patient demographics and clinical factors
- **Clinical Relevance**: Traditional clinical factors that influence cancer risk and prognosis

### 6. **🧬 ICGC Pathway Features (20 features)**
- **Range**: Features 90-109 (`icgc_0` to `icgc_19`)
- **Data Type**: Float values (0.0 - 5.0)
- **Description**: ICGC ARGO pathway data from international cancer genomics consortium
- **Clinical Relevance**: Pathway-level alterations from comprehensive genomic analysis

---

## 🔍 **SHAP Explainability Integration**

The Features tab displays comprehensive information about AI explainability support:

### SHAP Support Status
- **✅ Enabled**: SHAP explanations are available for predictions
- **Available Explainers**: 3 different explainer types for different model architectures
- **Explanation Methods**:
  - **TreeExplainer**: For Random Forest and Gradient Boosting models
  - **KernelExplainer**: For Deep Neural Network models (fallback)

### Benefits of SHAP Integration
- **Per-Feature Contributions**: See exactly how each feature influences predictions
- **Clinical Trust**: Transparent AI decision-making for clinical settings
- **Regulatory Compliance**: Explainable AI for medical device requirements
- **Research Insights**: Understand model behavior and feature importance

---

## 🎯 **Using the Features Tab**

### Navigation
1. **Open the Cancer Alpha Web App**: Navigate to `http://localhost:3000`
2. **Click the Features Tab**: Located in the main navigation bar
3. **Explore the Interface**: Use the filtering and information panels

### Interface Components

#### **Summary Statistics Cards**
- **Total Features**: Shows the complete count of 110 features
- **SHAP Support**: Displays explainability status (Yes/No)
- **Categories**: Number of feature categories (6)
- **Models**: Number of explainable model types (3)

#### **Category Filter**
- **Dropdown Selection**: Choose a specific category or "All Categories"
- **Interactive Chips**: Click category chips to filter the view
- **Category Count**: See how many features are in each category

#### **Category Detail Cards**
For each selected category, you'll see:
- **Description**: Detailed explanation of the feature type
- **Count & Range**: Number of features and their index range
- **Clinical Relevance**: Biological and medical significance
- **Example Features**: Sample feature names from the category

#### **Explainability Information Panel**
- **SHAP Support Status**: Current explainability capabilities
- **Available Explainers**: List of supported model explainers
- **Explanation Methods**: Technical details about explainer types

#### **Usage Notes Section**
- **Best Practices**: Guidelines for optimal feature usage
- **Data Requirements**: Technical specifications for feature input
- **Accuracy Considerations**: How feature completeness affects predictions

---

## 📊 **Technical Specifications**

### Feature Data Requirements
- **Total Features**: Exactly 110 features required for optimal accuracy
- **Missing Features**: Automatically zero-filled (may reduce accuracy)
- **Feature Scaling**: Automatically applied using trained scaler
- **SHAP Explanations**: Available for per-feature contribution analysis

### Expected Value Ranges

| **Feature Category** | **Expected Range** | **Data Type** | **Count** |
|----------------------|-------------------|---------------|-----------|
| Methylation | 0.0 - 1.0 | Float | 20 |
| Mutation | 0 - 50+ | Integer | 25 |
| Copy Number | 0.0 - 10.0 | Float | 20 |
| Fragmentomics | 50.0 - 500.0 | Float | 15 |
| Clinical | 0.0 - 1.0 | Float | 10 |
| ICGC Pathway | 0.0 - 5.0 | Float | 20 |

---

## 🔬 **Research Applications**

### For Researchers
- **Feature Selection**: Understand which feature categories are most relevant
- **Data Preparation**: Learn proper data formatting and range expectations
- **Model Interpretation**: Use SHAP explanations to understand model decisions
- **Biological Insights**: Connect feature categories to biological processes

### For Clinicians
- **Clinical Relevance**: Understand how different data types contribute to predictions
- **Trust Building**: See transparent explanations for AI predictions
- **Biomarker Discovery**: Identify important features for specific cancer types
- **Patient Stratification**: Use feature insights for personalized medicine

### For Data Scientists
- **Feature Engineering**: Guidelines for preparing genomic datasets
- **Model Validation**: Use SHAP explanations to validate model behavior
- **Performance Optimization**: Understand feature importance distributions
- **Integration Planning**: Technical specifications for system integration

---

## 🛠️ **Technical Integration**

### API Endpoint
The Features tab consumes data from the `/features/info` API endpoint:

```bash
curl -X GET "http://localhost:8001/features/info" \
     -H "accept: application/json"
```

### Response Structure
```json
{
  "total_features": 110,
  "feature_categories": {
    "methylation": {
      "count": 20,
      "range": "0-20",
      "description": "DNA methylation patterns...",
      "example_features": ["methylation_0", "methylation_1"],
      "clinical_relevance": "Hypermethylation often associated..."
    },
    // ... other categories
  },
  "explainability": {
    "shap_support": true,
    "available_explainers": ["random_forest", "gradient_boosting", "deep_neural_network"],
    "explanation_methods": {
      "TreeExplainer": "For Random Forest and Gradient Boosting models",
      "KernelExplainer": "For Deep Neural Network models (fallback)"
    }
  },
  "usage_notes": [
    "All 110 features should be provided for optimal prediction accuracy",
    "Missing features will be zero-filled (may reduce accuracy)",
    "Features are automatically scaled using trained scaler",
    "SHAP explanations show per-feature contributions to predictions"
  ]
}
```

---

## 🔧 **Troubleshooting**

### Common Issues

#### **Features Tab Not Loading**
- **Check API Connection**: Ensure the API is running on port 8001
- **Verify Endpoint**: Test `/features/info` endpoint manually
- **Network Issues**: Check for CORS or network connectivity problems

#### **Missing Category Information**
- **API Version**: Ensure you're using the latest API version (2.0.0+)
- **Model Loading**: Verify that all models have loaded successfully
- **Feature Data**: Check that feature metadata is available

#### **SHAP Information Not Displaying**
- **Model Compatibility**: Ensure SHAP explainers are initialized
- **Memory Issues**: Check available system memory for SHAP calculations
- **Model Loading**: Verify that explainer objects have been created

### Debug Commands
```bash
# Test API health
curl http://localhost:8001/health

# Test features endpoint
curl http://localhost:8001/features/info

# Check web app console
# Open browser developer tools and check for JavaScript errors
```

---

## 📚 **Additional Resources**

### Documentation
- **API Reference**: [API_REFERENCE_GUIDE.md](API_REFERENCE_GUIDE.md)
- **Master Installation Guide**: [MASTER_INSTALLATION_GUIDE.md](../MASTER_INSTALLATION_GUIDE.md)
- **Web App Deployment**: [Phase 4B Guide](../src/phase4_systemization_and_tool_deployment/docs/phase4b_guide.md)

### Related Features
- **Prediction Interface**: Make predictions with feature explanations
- **Results View**: See SHAP explanations for individual predictions
- **Model Management**: View model performance and capabilities

### Support
- **API Documentation**: http://localhost:8001/docs
- **Interactive Testing**: Use the Swagger UI for API testing

---

## ⚠️ **Important Notes**

### Research Use Only
The Features tab is designed for research and educational purposes. Clinical applications require additional validation and regulatory approval.

### Data Privacy
- **No Patient Data**: The Features tab displays general information only
- **Metadata Only**: No actual patient features or predictions are shown
- **Educational Purpose**: Designed for understanding feature categories, not individual analysis

### Technical Limitations
- **Feature Count**: System expects exactly 110 features as specified
- **Data Format**: Features must match expected ranges and data types
- **SHAP Availability**: Explanations depend on model type and system resources

---

**Last Updated:** July 25, 2025  
**Web App Version:** 2.0.0 - SHAP Explainability Enabled  
**Documentation Version:** 1.0
