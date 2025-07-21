"""
Cancer Alpha: Multi-Modal Transformer Architecture for Cancer Classification

A modern machine learning framework for cancer classification using 
multi-modal transformer architectures including TabTransformer, 
Multi-Modal Transformer, and Perceiver IO models.
"""

__version__ = "1.0.0"
__author__ = "Cancer Alpha Research Team"
__email__ = "research@cancer-alpha.org"

from . import data, models, transformers, visualization, utils

__all__ = ["data", "models", "transformers", "visualization", "utils"]
