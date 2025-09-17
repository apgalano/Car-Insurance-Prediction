"""Complete prediction pipeline for car insurance model."""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Union, List
import warnings

from .logistic_regression import CarInsuranceModel
from ..data.preprocessor import DataPreprocessor
from ..config.settings import MODELS_DIR


class CarInsurancePredictionPipeline:
    """Complete pipeline for car insurance predictions including preprocessing."""
    
    def __init__(self):
        self.model = CarInsuranceModel()
        self.preprocessor = DataPreprocessor()
        self.is_fitted = False
        self.feature_names = None
        
    def fit(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fit the complete pipeline on training data."""
        print("Fitting prediction pipeline...")
        
        # Preprocess the data
        X, y = self.preprocessor.preprocess_training_data(data)
        
        # Get feature names before scaling
        temp_X, temp_y = self.preprocessor.split_features_target(
            self.preprocessor.reorder_columns(
                self.preprocessor.create_dummy_variables(
                    self.preprocessor.handle_missing_values(
                        self.preprocessor.remove_outliers(
                            self.preprocessor.create_call_duration_feature(data).loc[0:3999, :].copy()
                        )
                    )
                )
            )
        )
        self.feature_names = list(temp_X.columns)
        
        # Train the model
        results = self.model.train(X, y, feature_names=self.feature_names)
        self.is_fitted = True
        
        return results
    
    def predict(self, data: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions")
        
        # Convert input to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list):
            data = pd.DataFrame(data)
        
        # Preprocess the new data using the same steps as training
        processed_data = self._preprocess_new_data(data)
        
        # Make predictions
        predictions = self.model.predict(processed_data)
        return predictions
    
    def predict_proba(self, data: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
        """Get prediction probabilities for new data."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions")
        
        # Convert input to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list):
            data = pd.DataFrame(data)
        
        # Preprocess the new data
        processed_data = self._preprocess_new_data(data)
        
        # Get probabilities
        probabilities = self.model.predict_proba(processed_data)
        return probabilities
    
    def _preprocess_new_data(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess new data for prediction using fitted preprocessor."""
        # Create a copy to avoid modifying original data
        data = data.copy()
        
        # Add missing columns with default values if needed
        required_columns = ['Age', 'Job', 'Marital', 'Education', 'Default', 'Balance', 
                          'HHInsurance', 'CarLoan', 'NoOfContacts', 'DaysPassed', 
                          'PrevAttempts', 'Communication', 'LastContactDay', 
                          'LastContactMonth', 'Outcome', 'CallStart', 'CallEnd']
        
        for col in required_columns:
            if col not in data.columns:
                # Set reasonable defaults for missing columns
                if col in ['Age', 'Balance', 'NoOfContacts', 'DaysPassed', 'PrevAttempts']:
                    data[col] = 0
                elif col in ['CallStart', 'CallEnd']:
                    data[col] = '00:00:00'
                else:
                    data[col] = 'unknown'
        
        # Apply the same preprocessing steps as training
        # 1. Create call duration feature
        data = self.preprocessor.create_call_duration_feature(data)
        
        # 2. Handle missing values
        data = self.preprocessor.handle_missing_values(data)
        
        # 3. Create dummy variables
        data = self.preprocessor.create_dummy_variables(data)
        
        # 4. Reorder columns to match training data
        data = self.preprocessor.reorder_columns(data)
        
        # 5. Ensure all training features are present (add missing dummy columns)
        for feature in self.feature_names:
            if feature not in data.columns:
                data[feature] = 0
        
        # 6. Select only the features used in training (in correct order)
        data = data[self.feature_names]
        
        # 7. Convert all columns to numeric (in case any string values remain)
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        
        # 8. Scale the features using the fitted scaler
        if not self.preprocessor.is_fitted:
            raise ValueError("Preprocessor scaler is not fitted")
        
        scaled_data = self.preprocessor.scaler.transform(data)
        
        return scaled_data
    
    def save_pipeline(self, filepath: Path = None) -> Path:
        """Save the complete pipeline (model + preprocessor) to disk."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")
        
        if filepath is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            filepath = MODELS_DIR / 'car_insurance_pipeline.pkl'
        
        # Save the entire pipeline
        pipeline_data = {
            'model': self.model.model,
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(pipeline_data, filepath)
        return filepath
    
    def load_pipeline(self, filepath: Path) -> None:
        """Load a complete pipeline from disk."""
        pipeline_data = joblib.load(filepath)
        
        self.model.model = pipeline_data['model']
        self.model.is_trained = True
        self.preprocessor = pipeline_data['preprocessor']
        self.feature_names = pipeline_data['feature_names']
        self.is_fitted = pipeline_data['is_fitted']
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the trained model."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before getting feature importance")
        
        return self.model.get_feature_importance()
    
    def predict_single(self, **kwargs) -> Dict[str, Any]:
        """Make a prediction for a single customer with named parameters."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions")
        
        # Create DataFrame from keyword arguments
        data = pd.DataFrame([kwargs])
        
        # Get prediction and probability
        prediction = self.predict(data)[0]
        probabilities = self.predict_proba(data)[0]
        
        return {
            'prediction': int(prediction),
            'will_buy_insurance': bool(prediction),
            'probability_no': float(probabilities[0]),
            'probability_yes': float(probabilities[1]),
            'confidence': float(max(probabilities))
        }