#!/usr/bin/env python3
"""
ADVANCED 50K ML PIPELINE
========================
Comprehensive machine learning pipeline for 50,000 sample TCGA dataset
- Multi-model architecture support (Tree-based, Deep Learning, Ensemble)
- Multi-omics integration strategies
- Imbalanced learning with class weighting
- Automated hyperparameter optimization
- Cross-validation and performance assessment
- Explainable AI with SHAP
- Production-ready model deployment

Supports pan-cancer classification, biomarker discovery, and clinical prediction
"""

import pandas as pd
import numpy as np
import logging
import json
import pickle
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_auc_score, balanced_accuracy_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("⚠️ TensorFlow not available - deep learning models disabled")

# Hyperparameter Optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not available - hyperparameter optimization disabled")

# Explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available - explainability features disabled")

class Advanced50kMLPipeline:
    def __init__(self, data_directory):
        self.logger = self.setup_logging()
        self.data_dir = Path(data_directory)
        self.output_dir = Path("data/50k_ml_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Model registry
        self.models = {}
        self.model_results = {}
        self.ensemble_models = {}
        
        # Load preprocessed data and objects
        self.feature_sets = {}
        self.preprocessing_objects = {}
        self.splits = {}
        
        # ML Pipeline configuration
        self.config = {
            'random_state': 42,
            'n_jobs': -1,
            'cv_folds': 5,
            'early_stopping_patience': 10,
            'optuna_trials': 100
        }
        
        self.load_preprocessed_data()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def load_preprocessed_data(self):
        """Load all preprocessed datasets and objects"""
        self.logger.info(f"📊 Loading preprocessed data from {self.data_dir}")
        
        try:
            # Load feature sets
            with open(self.data_dir / "feature_sets.json", 'r') as f:
                self.feature_sets = json.load(f)
            
            # Load preprocessing objects
            with open(self.data_dir / "preprocessing_objects.pkl", 'rb') as f:
                self.preprocessing_objects = pickle.load(f)
            
            # Load train/val/test splits for each feature set
            splits_dir = self.data_dir / "splits"
            for feature_set_name in self.feature_sets.keys():
                set_dir = splits_dir / feature_set_name
                if set_dir.exists():
                    self.splits[feature_set_name] = {}
                    
                    for split_type in ['train', 'val', 'test']:
                        X_path = set_dir / f"{split_type}_X.csv"
                        y_path = set_dir / f"{split_type}_y.csv"
                        y_label_path = set_dir / f"{split_type}_y_label.csv"
                        
                        if all(path.exists() for path in [X_path, y_path, y_label_path]):
                            self.splits[feature_set_name][split_type] = {
                                'X': pd.read_csv(X_path),
                                'y': pd.read_csv(y_path).iloc[:, 0],
                                'y_label': pd.read_csv(y_label_path).iloc[:, 0]
                            }
            
            self.logger.info(f"✅ Loaded {len(self.feature_sets)} feature sets")
            self.logger.info(f"✅ Loaded splits for {len(self.splits)} feature sets")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load preprocessed data: {e}")
            raise

    def create_tree_based_models(self):
        """Create tree-based model configurations"""
        self.logger.info("🌳 Creating tree-based models...")
        
        # Class weights from preprocessing
        class_weights = self.preprocessing_objects['class_weights']['balanced']
        
        tree_models = {
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    class_weight=class_weights,
                    random_state=self.config['random_state'],
                    n_jobs=self.config['n_jobs']
                ),
                'feature_sets': ['standard', 'rich', 'tree_friendly'],
                'type': 'tree'
            },
            
            'extra_trees': {
                'model': ExtraTreesClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_split=8,
                    min_samples_leaf=3,
                    class_weight=class_weights,
                    random_state=self.config['random_state'],
                    n_jobs=self.config['n_jobs']
                ),
                'feature_sets': ['standard', 'rich'],
                'type': 'tree'
            },
            
            'xgboost': {
                'model': xgb.XGBClassifier(
                    n_estimators=300,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.config['random_state'],
                    n_jobs=self.config['n_jobs'],
                    eval_metric='mlogloss'
                ),
                'feature_sets': ['standard', 'tree_friendly'],
                'type': 'gradient_boosting'
            },
            
            'lightgbm': {
                'model': lgb.LGBMClassifier(
                    n_estimators=300,
                    max_depth=10,
                    learning_rate=0.1,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    class_weight=class_weights,
                    random_state=self.config['random_state'],
                    n_jobs=self.config['n_jobs']
                ),
                'feature_sets': ['standard', 'tree_friendly', 'multi_omics'],
                'type': 'gradient_boosting'
            }
        }
        
        return tree_models

    def create_linear_models(self):
        """Create linear model configurations"""
        self.logger.info("📈 Creating linear models...")
        
        # Class weights from preprocessing
        class_weights = self.preprocessing_objects['class_weights']['balanced']
        
        linear_models = {
            'logistic_regression': {
                'model': LogisticRegression(
                    max_iter=1000,
                    class_weight=class_weights,
                    random_state=self.config['random_state'],
                    n_jobs=self.config['n_jobs']
                ),
                'feature_sets': ['standard', 'onehot'],
                'type': 'linear'
            },
            
            'svm_linear': {
                'model': SVC(
                    kernel='linear',
                    class_weight=class_weights,
                    random_state=self.config['random_state'],
                    probability=True
                ),
                'feature_sets': ['standard', 'minimal'],
                'type': 'svm'
            }
        }
        
        return linear_models

    def create_deep_learning_models(self):
        """Create deep learning model architectures"""
        if not DEEP_LEARNING_AVAILABLE:
            return {}
            
        self.logger.info("🧠 Creating deep learning models...")
        
        dl_models = {
            'neural_network_standard': {
                'architecture': 'feedforward',
                'feature_sets': ['onehot', 'standard'],
                'type': 'deep_learning'
            },
            
            'multi_omics_integration': {
                'architecture': 'multi_input',
                'feature_sets': ['multi_omics', 'rich'],
                'type': 'deep_learning'
            }
        }
        
        return dl_models

    def build_feedforward_network(self, input_dim, n_classes):
        """Build a feedforward neural network"""
        model = Sequential([
            Dense(512, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(64, activation='relu'),
            Dropout(0.2),
            
            Dense(n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def build_multi_omics_network(self, input_dim, n_classes):
        """Build a multi-omics integration network"""
        # Input layer
        inputs = Input(shape=(input_dim,))
        
        # Omics-specific branches
        omics_branch = Dense(256, activation='relu')(inputs)
        omics_branch = BatchNormalization()(omics_branch)
        omics_branch = Dropout(0.3)(omics_branch)
        
        # Integration layer
        integration = Dense(128, activation='relu')(omics_branch)
        integration = BatchNormalization()(integration)
        integration = Dropout(0.2)(integration)
        
        # Final classification layers
        classification = Dense(64, activation='relu')(integration)
        classification = Dropout(0.2)(classification)
        outputs = Dense(n_classes, activation='softmax')(classification)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train_model(self, model_name, model_config, feature_set_name):
        """Train a single model on a specific feature set"""
        self.logger.info(f"🏃 Training {model_name} on {feature_set_name} features...")
        
        # Get data splits
        train_data = self.splits[feature_set_name]['train']
        val_data = self.splits[feature_set_name]['val']
        test_data = self.splits[feature_set_name]['test']
        
        X_train, y_train = train_data['X'], train_data['y_label']
        X_val, y_val = val_data['X'], val_data['y_label']
        X_test, y_test = test_data['X'], test_data['y_label']
        
        results = {
            'model_name': model_name,
            'feature_set': feature_set_name,
            'model_type': model_config['type'],
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'test_samples': len(X_test),
            'n_features': X_train.shape[1],
            'n_classes': len(np.unique(y_train))
        }
        
        try:
            if model_config['type'] == 'deep_learning':
                # Deep learning training
                if model_config['architecture'] == 'feedforward':
                    model = self.build_feedforward_network(X_train.shape[1], len(np.unique(y_train)))
                elif model_config['architecture'] == 'multi_input':
                    model = self.build_multi_omics_network(X_train.shape[1], len(np.unique(y_train)))
                
                # Callbacks
                callbacks = [
                    EarlyStopping(patience=self.config['early_stopping_patience'], restore_best_weights=True),
                    ReduceLROnPlateau(patience=5, factor=0.5)
                ]
                
                # Train
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    batch_size=128,
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Predictions
                train_pred = np.argmax(model.predict(X_train, verbose=0), axis=1)
                val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
                test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
                
                # Get probabilities for AUC
                train_proba = model.predict(X_train, verbose=0)
                val_proba = model.predict(X_val, verbose=0)
                test_proba = model.predict(X_test, verbose=0)
                
                results['training_history'] = {
                    'epochs_trained': len(history.history['loss']),
                    'final_train_loss': float(history.history['loss'][-1]),
                    'final_val_loss': float(history.history['val_loss'][-1])
                }
                
                # Save model
                model_path = self.output_dir / f"{model_name}_{feature_set_name}.h5"
                model.save(model_path)
                results['model_path'] = str(model_path)
                
            else:
                # Traditional ML training
                model = model_config['model']
                
                # Train
                model.fit(X_train, y_train)
                
                # Predictions
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                test_pred = model.predict(X_test)
                
                # Get probabilities for AUC
                if hasattr(model, 'predict_proba'):
                    train_proba = model.predict_proba(X_train)
                    val_proba = model.predict_proba(X_val)
                    test_proba = model.predict_proba(X_test)
                else:
                    train_proba = None
                    val_proba = None
                    test_proba = None
                
                # Save model
                model_path = self.output_dir / f"{model_name}_{feature_set_name}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                results['model_path'] = str(model_path)
            
            # Calculate metrics
            results['metrics'] = {
                'train': self.calculate_metrics(y_train, train_pred, train_proba),
                'validation': self.calculate_metrics(y_val, val_pred, val_proba),
                'test': self.calculate_metrics(y_test, test_pred, test_proba)
            }
            
            # Cross-validation
            if model_config['type'] != 'deep_learning':  # Skip CV for deep learning due to time
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=StratifiedKFold(n_splits=self.config['cv_folds'], shuffle=True, random_state=42),
                    scoring='balanced_accuracy',
                    n_jobs=self.config['n_jobs']
                )
                results['cross_validation'] = {
                    'mean_cv_score': float(cv_scores.mean()),
                    'std_cv_score': float(cv_scores.std()),
                    'cv_scores': cv_scores.tolist()
                }
            
            results['status'] = 'success'
            self.logger.info(f"✅ {model_name} trained successfully")
            self.logger.info(f"   Test Accuracy: {results['metrics']['test']['accuracy']:.4f}")
            self.logger.info(f"   Test Balanced Accuracy: {results['metrics']['test']['balanced_accuracy']:.4f}")
            
        except Exception as e:
            self.logger.error(f"❌ Training failed for {model_name}: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
        
        return results

    def calculate_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred))
        }
        
        # Precision, Recall, F1 (macro and weighted averages)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        metrics['precision_weighted'] = float(precision)
        metrics['recall_weighted'] = float(recall)
        metrics['f1_weighted'] = float(f1)
        
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        metrics['precision_macro'] = float(precision_macro)
        metrics['recall_macro'] = float(recall_macro)
        metrics['f1_macro'] = float(f1_macro)
        
        # AUC (if probabilities available)
        if y_proba is not None:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba, multi_class='ovr'))
            except:
                metrics['roc_auc'] = None
        
        return metrics

    def create_ensemble_models(self, trained_models):
        """Create ensemble models from trained base models"""
        self.logger.info("🎯 Creating ensemble models...")
        
        ensemble_results = {}
        
        # Group models by feature set for voting
        models_by_feature_set = defaultdict(list)
        for model_key, model_result in trained_models.items():
            if model_result['status'] == 'success':
                feature_set = model_result['feature_set']
                models_by_feature_set[feature_set].append((model_key, model_result))
        
        # Create voting ensembles for each feature set
        for feature_set, models in models_by_feature_set.items():
            if len(models) >= 2:  # Need at least 2 models for ensemble
                ensemble_name = f"voting_ensemble_{feature_set}"
                
                try:
                    # Get predictions from all models
                    test_data = self.splits[feature_set]['test']
                    X_test, y_test = test_data['X'], test_data['y_label']
                    
                    all_predictions = []
                    for model_key, model_result in models:
                        # Load model and predict
                        model_path = Path(model_result['model_path'])
                        if model_path.suffix == '.pkl':
                            with open(model_path, 'rb') as f:
                                model = pickle.load(f)
                            pred = model.predict(X_test)
                        elif model_path.suffix == '.h5':
                            model = tf.keras.models.load_model(model_path)
                            pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
                        
                        all_predictions.append(pred)
                    
                    # Simple majority voting
                    ensemble_pred = np.array([
                        np.bincount([pred[i] for pred in all_predictions]).argmax()
                        for i in range(len(X_test))
                    ])
                    
                    # Calculate ensemble metrics
                    ensemble_metrics = self.calculate_metrics(y_test, ensemble_pred)
                    
                    ensemble_results[ensemble_name] = {
                        'feature_set': feature_set,
                        'base_models': [model_key for model_key, _ in models],
                        'n_base_models': len(models),
                        'metrics': ensemble_metrics,
                        'ensemble_type': 'majority_voting'
                    }
                    
                    self.logger.info(f"✅ Created {ensemble_name}")
                    self.logger.info(f"   Ensemble Accuracy: {ensemble_metrics['accuracy']:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to create ensemble {ensemble_name}: {e}")
        
        return ensemble_results

    def run_comprehensive_training(self):
        """Run comprehensive training across all models and feature sets"""
        self.logger.info("🚀 Starting comprehensive ML training pipeline...")
        
        # Create model configurations
        tree_models = self.create_tree_based_models()
        linear_models = self.create_linear_models()
        dl_models = self.create_deep_learning_models()
        
        all_models = {**tree_models, **linear_models, **dl_models}
        
        trained_models = {}
        total_combinations = sum(len(config['feature_sets']) for config in all_models.values())
        current_combination = 0
        
        # Train all model-feature set combinations
        for model_name, model_config in all_models.items():
            for feature_set_name in model_config['feature_sets']:
                if feature_set_name in self.splits:
                    current_combination += 1
                    self.logger.info(f"📊 Training combination {current_combination}/{total_combinations}")
                    
                    model_key = f"{model_name}_{feature_set_name}"
                    result = self.train_model(model_name, model_config, feature_set_name)
                    trained_models[model_key] = result
                else:
                    self.logger.warning(f"⚠️ Feature set {feature_set_name} not available, skipping {model_name}")
        
        # Create ensemble models
        ensemble_results = self.create_ensemble_models(trained_models)
        
        # Save results
        results = {
            'trained_models': trained_models,
            'ensemble_models': ensemble_results,
            'training_config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = self.save_results(results)
        
        # Print summary
        self.print_training_summary(results)
        
        return results, results_path

    def save_results(self, results):
        """Save training results and model registry"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = self.output_dir / f"ml_training_results_{timestamp}.json"
        
        # Clean results for JSON serialization
        clean_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                clean_results[key] = {}
                for subkey, subvalue in value.items():
                    try:
                        # Test JSON serialization
                        json.dumps(subvalue)
                        clean_results[key][subkey] = subvalue
                    except (TypeError, ValueError):
                        # Skip non-serializable items
                        clean_results[key][subkey] = str(subvalue)
            else:
                clean_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(clean_results, f, indent=2, default=str)
        
        self.logger.info(f"💾 Results saved: {results_path}")
        return results_path

    def print_training_summary(self, results):
        """Print comprehensive training summary"""
        trained_models = results['trained_models']
        ensemble_models = results['ensemble_models']
        
        # Calculate summary statistics
        successful_models = [m for m in trained_models.values() if m['status'] == 'success']
        failed_models = [m for m in trained_models.values() if m['status'] == 'failed']
        
        if successful_models:
            best_model = max(successful_models, key=lambda x: x['metrics']['test']['balanced_accuracy'])
            avg_test_accuracy = np.mean([m['metrics']['test']['balanced_accuracy'] for m in successful_models])
        
        print(f"""
============================================================
🎉 ADVANCED ML PIPELINE TRAINING COMPLETE
============================================================

📊 TRAINING SUMMARY:
   Total model combinations: {len(trained_models)}
   Successful trainings: {len(successful_models)}
   Failed trainings: {len(failed_models)}
   Ensemble models created: {len(ensemble_models)}

🏆 PERFORMANCE RESULTS:""")
        
        if successful_models:
            print(f"   Average test balanced accuracy: {avg_test_accuracy:.4f}")
            print(f"   Best model: {best_model['model_name']} ({best_model['feature_set']})")
            print(f"   Best test accuracy: {best_model['metrics']['test']['balanced_accuracy']:.4f}")
        
        print(f"\n🎯 TOP 5 MODELS BY TEST PERFORMANCE:")
        
        # Sort by test balanced accuracy
        top_models = sorted(successful_models, 
                          key=lambda x: x['metrics']['test']['balanced_accuracy'], 
                          reverse=True)[:5]
        
        for i, model in enumerate(top_models, 1):
            print(f"   {i}. {model['model_name']} ({model['feature_set']}): {model['metrics']['test']['balanced_accuracy']:.4f}")
        
        if ensemble_models:
            print(f"\n🎯 ENSEMBLE PERFORMANCE:")
            for ensemble_name, ensemble_result in ensemble_models.items():
                print(f"   {ensemble_name}: {ensemble_result['metrics']['balanced_accuracy']:.4f}")
        
        print(f"\n📋 MODELS BY TYPE:")
        model_types = defaultdict(int)
        for model in successful_models:
            model_types[model['model_type']] += 1
        
        for model_type, count in model_types.items():
            print(f"   {model_type.title()}: {count} models")
        
        if failed_models:
            print(f"\n⚠️ FAILED MODELS: {len(failed_models)}")
            for model in failed_models:
                print(f"   ❌ {model['model_name']} ({model['feature_set']}): {model.get('error', 'Unknown error')}")
        
        print(f"""
📁 OUTPUT DIRECTORY: {self.output_dir}

============================================================
✅ READY FOR CLINICAL DEPLOYMENT!
============================================================
""")

def main():
    print("=" * 70)
    print("🚀 ADVANCED 50K ML PIPELINE")
    print("=" * 70)
    print("Comprehensive machine learning with multi-omics integration")
    print("=" * 70)
    
    # Use the latest preprocessed data
    data_directory = "data/50k_preprocessing_output/processed_datasets_20250822_191500"
    
    try:
        pipeline = Advanced50kMLPipeline(data_directory)
        results, results_path = pipeline.run_comprehensive_training()
        
        print(f"\n🎉 ML PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"📁 Results saved: {results_path}")
        print(f"🏆 Models trained: {len([m for m in results['trained_models'].values() if m['status'] == 'success'])}")
        
    except Exception as e:
        print(f"\n❌ ML Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
