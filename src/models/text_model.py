"""
Lifestyle Risk Model for Student Depression Dataset
Predicts risk based on lifestyle and mental health indicators
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


class LifestyleRiskModel:
    """Model for predicting lifestyle and depression risk"""
    
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
        self.is_trained = True
    
    def predict(self, X):
        """Make predictions on new data"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.predict_proba(X)
