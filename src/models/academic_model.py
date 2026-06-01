"""
Academic Risk Model for OULAD Dataset
Predicts student risk based on academic performance data
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class AcademicRiskModel:
    """Model for predicting academic risk from OULAD dataset"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
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
