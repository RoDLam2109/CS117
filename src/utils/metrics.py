"""
Metrics Calculation Utilities
Provides functions for calculating model performance metrics
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_auc_score
)


def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate comprehensive performance metrics
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_proba : array-like, optional
        Predicted probabilities for ROC-AUC
    
    Returns:
    --------
    dict : Dictionary containing metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    
    return metrics


def print_metrics(metrics):
    """Print metrics in a readable format"""
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
