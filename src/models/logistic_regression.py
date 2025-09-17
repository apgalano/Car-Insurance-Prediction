"""Logistic regression model implementation."""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from typing import Tuple, Dict, Any
import joblib
from pathlib import Path

from ..config.settings import RANDOM_STATE, TEST_SIZE, CV_FOLDS, MODELS_DIR


class CarInsuranceModel:
    """Logistic regression model for car insurance prediction."""
    
    def __init__(self, cv_folds: int = CV_FOLDS, random_state: int = RANDOM_STATE):
        self.model = LogisticRegressionCV(cv=cv_folds, random_state=random_state)
        self.feature_names = None
        self.is_trained = False
        self.cv_folds = cv_folds
        self.random_state = random_state
    
    def train(self, X: np.ndarray, y: pd.Series, feature_names: list = None) -> Dict[str, Any]:
        """Train the logistic regression model."""
        self.feature_names = feature_names
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=self.random_state, shuffle=True
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate accuracies
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        
        # Generate predictions and classification report
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred)
        
        results = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': report,
            'model_coefficients': self.model.coef_[0],
            'model_intercept': self.model.intercept_[0]
        }
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance based on model coefficients."""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if self.feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.model.coef_[0]))]
        else:
            feature_names = self.feature_names
        
        # Create summary table
        summary_table = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': self.model.coef_[0]
        })
        
        # Add intercept
        intercept_row = pd.DataFrame({
            'Feature': ['Intercept'],
            'Coefficient': [self.model.intercept_[0]]
        })
        
        summary_table = pd.concat([intercept_row, summary_table], ignore_index=True)
        summary_table = summary_table.sort_values('Coefficient', ascending=False)
        summary_table.reset_index(drop=True, inplace=True)
        
        return summary_table
    
    # Note: Individual model saving/loading removed - use CarInsurancePredictionPipeline instead
    # The pipeline contains both the model and preprocessor needed for predictions
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of the model parameters and performance."""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting summary")
        
        return {
            'cv_folds': self.cv_folds,
            'random_state': self.random_state,
            'n_features': len(self.model.coef_[0]),
            'intercept': self.model.intercept_[0],
            'regularization_path': len(self.model.Cs_)
        }