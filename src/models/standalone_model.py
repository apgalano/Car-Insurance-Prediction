"""Standalone model that can be used without the source code."""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


class StandaloneCarInsurancePredictor:
    """
    Standalone car insurance predictor that can be used without source code.
    
    This class contains all preprocessing logic and the trained model
    in a single, self-contained class that can be pickled and used
    in external repositories.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
        # Configuration constants
        self.MISSING_VALUE_REPLACEMENTS = {
            'Job': 'management',
            'Education': 'tertiary',
            'Communication': 'cellular',
            'Outcome': 'other'
        }
        
        self.CATEGORICAL_COLUMNS = [
            'Job', 'Marital', 'Education', 'Communication', 
            'LastContactDay', 'LastContactMonth', 'Outcome'
        ]
        
        self.BALANCE_THRESHOLD = 70000
        self.PREV_ATTEMPTS_THRESHOLD = 35
    
    def _create_call_duration_feature(self, data):
        """Create CallDuration feature from CallStart and CallEnd."""
        data = data.copy()
        frmt = '%H:%M:%S'
        call_durations = []
        
        for _, row in data.iterrows():
            try:
                t1 = datetime.strptime(row['CallStart'], frmt)
                t2 = datetime.strptime(row['CallEnd'], frmt)
                dt = t2 - t1
                call_durations.append(dt.total_seconds())
            except:
                call_durations.append(300)  # Default 5 minutes
        
        data['CallDuration'] = call_durations
        
        # Drop columns that exist
        columns_to_drop = ['CallStart', 'CallEnd']
        if 'Id' in data.columns:
            columns_to_drop.append('Id')
        
        data = data.drop(columns_to_drop, axis=1, errors='ignore')
        return data
    
    def _remove_outliers(self, data):
        """Remove outliers based on predefined thresholds."""
        data = data.copy()
        
        # Remove balance outliers
        if 'Balance' in data.columns:
            data = data[data['Balance'] <= self.BALANCE_THRESHOLD]
        
        # Remove previous attempts outliers
        if 'PrevAttempts' in data.columns:
            data = data[data['PrevAttempts'] <= self.PREV_ATTEMPTS_THRESHOLD]
        
        return data
    
    def _handle_missing_values(self, data):
        """Handle missing values using predefined replacements."""
        data = data.copy()
        
        for column, replacement in self.MISSING_VALUE_REPLACEMENTS.items():
            if column in data.columns:
                data[column] = data[column].fillna(replacement)
        
        return data
    
    def _create_dummy_variables(self, data):
        """Create dummy variables for categorical features."""
        data = data.copy()
        
        # Create dummy variables for each categorical column
        dummy_dfs = []
        for col in self.CATEGORICAL_COLUMNS:
            if col in data.columns:
                dummy_df = pd.get_dummies(data[col])
                dummy_dfs.append(dummy_df)
        
        # Join all dummy variables
        for dummy_df in dummy_dfs:
            data = data.join(dummy_df, rsuffix='_dup')
        
        # Drop original categorical columns
        data.drop(self.CATEGORICAL_COLUMNS, axis=1, inplace=True, errors='ignore')
        
        # Drop any duplicate columns that were created
        duplicate_cols = [col for col in data.columns if str(col).endswith('_dup')]
        data.drop(duplicate_cols, axis=1, inplace=True, errors='ignore')
        
        return data
    
    def _preprocess_data(self, data):
        """Complete preprocessing pipeline."""
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
        
        # Apply preprocessing steps
        data = self._create_call_duration_feature(data)
        data = self._handle_missing_values(data)
        data = self._create_dummy_variables(data)
        
        # Convert all column names to strings
        data.columns = data.columns.astype(str)
        
        # Ensure all training features are present (add missing dummy columns)
        if self.feature_names:
            for feature in self.feature_names:
                if feature not in data.columns:
                    data[feature] = 0
            
            # Select only the features used in training (in correct order)
            data = data[self.feature_names]
        
        # Convert all columns to numeric (in case any string values remain)
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        
        return data
    
    def fit(self, model, scaler, feature_names):
        """Fit the standalone predictor with trained components."""
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.is_fitted = True
    
    def predict(self, data):
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Predictor must be fitted before making predictions")
        
        # Convert input to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list):
            data = pd.DataFrame(data)
        
        # Preprocess the data
        processed_data = self._preprocess_data(data)
        
        # Scale the features
        scaled_data = self.scaler.transform(processed_data)
        
        # Make predictions
        predictions = self.model.predict(scaled_data)
        return predictions
    
    def predict_proba(self, data):
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Predictor must be fitted before making predictions")
        
        # Convert input to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list):
            data = pd.DataFrame(data)
        
        # Preprocess the data
        processed_data = self._preprocess_data(data)
        
        # Scale the features
        scaled_data = self.scaler.transform(processed_data)
        
        # Get probabilities
        probabilities = self.model.predict_proba(scaled_data)
        return probabilities
    
    def predict_single(self, **kwargs):
        """Make a prediction for a single customer with named parameters."""
        if not self.is_fitted:
            raise ValueError("Predictor must be fitted before making predictions")
        
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


def create_standalone_model():
    """Create a standalone model from the existing pipeline."""
    from .prediction_pipeline import CarInsurancePredictionPipeline
    from ..config.settings import MODELS_DIR
    
    # Load the existing pipeline
    pipeline = CarInsurancePredictionPipeline()
    pipeline.load_pipeline(MODELS_DIR / 'car_insurance_pipeline.pkl')
    
    # Create standalone predictor
    standalone = StandaloneCarInsurancePredictor()
    standalone.fit(
        model=pipeline.model.model,
        scaler=pipeline.preprocessor.scaler,
        feature_names=pipeline.feature_names
    )
    
    return standalone