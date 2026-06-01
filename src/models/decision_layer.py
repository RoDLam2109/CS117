"""
Decision Layer Model
Combines predictions from all three modules into final risk decision
"""

import pandas as pd
import numpy as np


class DecisionLayerModel:
    """Synthetic decision layer combining all module predictions"""
    
    def __init__(self, weights=None):
        """
        Initialize decision layer with optional module weights
        
        Parameters:
        -----------
        weights : dict, optional
            Weights for each module (academic, lifestyle, text)
            Default: equal weights (1/3 each)
        """
        if weights is None:
            self.weights = {
                'academic': 1/3,
                'lifestyle': 1/3,
                'text': 1/3
            }
        else:
            self.weights = weights
    
    def combine_predictions(self, academic_risk, lifestyle_risk, text_risk):
        """
        Combine predictions from all three modules
        
        Parameters:
        -----------
        academic_risk : array-like
            Risk scores from academic module (0-1)
        lifestyle_risk : array-like
            Risk scores from lifestyle module (0-1)
        text_risk : array-like
            Risk scores from text module (0-1)
        
        Returns:
        --------
        array-like : Combined risk scores
        """
        combined = (
            academic_risk * self.weights['academic'] +
            lifestyle_risk * self.weights['lifestyle'] +
            text_risk * self.weights['text']
        )
        return combined
    
    def make_decision(self, combined_risk, threshold=0.5):
        """
        Make final risk decision based on combined score
        
        Parameters:
        -----------
        combined_risk : array-like
            Combined risk scores
        threshold : float, default=0.5
            Threshold for risk classification
        
        Returns:
        --------
        array : Binary risk classification (0=no risk, 1=at risk)
        """
        return (combined_risk >= threshold).astype(int)
