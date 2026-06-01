"""
Explainability Utilities
Provides functions for model interpretation and explainability
"""

import pandas as pd
import numpy as np


def explain_predictions(model, X, feature_names=None, top_n=5):
    """
    Explain model predictions using feature importance
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_ attribute
    X : array-like
        Input features
    feature_names : list, optional
        Names of features for better readability
    top_n : int, default=5
        Number of top important features to return
    
    Returns:
    --------
    pd.DataFrame : Top N important features with importance scores
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model must have feature_importances_ attribute")
    
    importances = model.feature_importances_
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(importances))]
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return feature_importance_df.head(top_n)


def get_decision_path(prediction_score, threshold=0.5, module_name=""):
    """
    Provide human-readable decision explanation
    
    Parameters:
    -----------
    prediction_score : float
        Model prediction score (0-1)
    threshold : float, default=0.5
        Risk classification threshold
    module_name : str, optional
        Name of the module for context
    
    Returns:
    --------
    str : Human-readable explanation
    """
    risk_level = "At Risk" if prediction_score >= threshold else "Low Risk"
    confidence = abs(prediction_score - 0.5) * 2  # Normalize to 0-1 scale
    
    explanation = f"{module_name} - Risk Level: {risk_level} "
    explanation += f"(Score: {prediction_score:.2f}, Confidence: {confidence:.2%})"
    
    return explanation
