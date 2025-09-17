"""Data preprocessing utilities."""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

from ..config.settings import (
    BALANCE_THRESHOLD, PREV_ATTEMPTS_THRESHOLD, 
    MISSING_VALUE_REPLACEMENTS, CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS_ORDER
)


class DataPreprocessor:
    """Handles data preprocessing and feature engineering."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def create_call_duration_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create CallDuration feature from CallStart and CallEnd."""
        data = data.copy()
        frmt = '%H:%M:%S'
        call_durations = []
        
        for _, row in data.iterrows():
            t1 = datetime.strptime(row['CallStart'], frmt)
            t2 = datetime.strptime(row['CallEnd'], frmt)
            dt = t2 - t1
            call_durations.append(dt.total_seconds())
        
        data['CallDuration'] = call_durations
        
        # Drop columns that exist (Id might not exist in new data)
        columns_to_drop = ['CallStart', 'CallEnd']
        if 'Id' in data.columns:
            columns_to_drop.append('Id')
        
        data = data.drop(columns_to_drop, axis=1)
        
        return data
    
    def remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers based on predefined thresholds."""
        data = data.copy()
        
        # Remove balance outliers
        data = data[data['Balance'] <= BALANCE_THRESHOLD]
        
        # Remove previous attempts outliers
        data = data[data['PrevAttempts'] <= PREV_ATTEMPTS_THRESHOLD]
        
        return data
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using predefined replacements."""
        data = data.copy()
        
        for column, replacement in MISSING_VALUE_REPLACEMENTS.items():
            if column in data.columns:
                data[column] = data[column].fillna(replacement)
        
        return data
    
    def create_dummy_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create dummy variables for categorical features."""
        data = data.copy()
        
        # Create dummy variables for each categorical column
        dummy_dfs = []
        for col in CATEGORICAL_COLUMNS:
            if col in data.columns:
                dummy_df = pd.get_dummies(data[col])
                dummy_dfs.append(dummy_df)
        
        # Join all dummy variables
        for dummy_df in dummy_dfs:
            data = data.join(dummy_df, rsuffix='_dup')
        
        # Drop original categorical columns
        data.drop(CATEGORICAL_COLUMNS, axis=1, inplace=True, errors='ignore')
        
        # Drop any duplicate columns that were created
        duplicate_cols = [col for col in data.columns if str(col).endswith('_dup')]
        data.drop(duplicate_cols, axis=1, inplace=True, errors='ignore')
        
        return data
    
    def reorder_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Reorder columns according to predefined order."""
        data = data.copy()
        
        # Convert all column names to strings to avoid mixed types
        data.columns = data.columns.astype(str)
        
        # Convert FEATURE_COLUMNS_ORDER to strings as well
        feature_columns_str = [str(col) for col in FEATURE_COLUMNS_ORDER]
        
        # Only reorder columns that exist in the data
        available_columns = [col for col in feature_columns_str if col in data.columns]
        
        return data[available_columns]
    
    def split_features_target(self, data: pd.DataFrame, target_col: str = 'CarInsurance') -> Tuple[pd.DataFrame, pd.Series]:
        """Split data into features and target."""
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        features = data.drop(target_col, axis=1)
        target = data[target_col]
        
        return features, target
    
    def scale_features(self, X_train: pd.DataFrame, X_test: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Scale features using StandardScaler."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.is_fitted = True
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled, None
    
    def preprocess_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
        """Complete preprocessing pipeline for training data."""
        # Feature engineering
        data = self.create_call_duration_feature(data)
        
        # Get only training data (first 4000 samples)
        train_data = data.loc[0:3999, :].copy()
        
        # Remove outliers
        train_data = self.remove_outliers(train_data)
        
        # Handle missing values
        train_data = self.handle_missing_values(train_data)
        
        # Create dummy variables
        train_data = self.create_dummy_variables(train_data)
        
        # Reorder columns
        train_data = self.reorder_columns(train_data)
        
        # Split features and target
        X, y = self.split_features_target(train_data)
        
        # Scale features
        X_scaled, _ = self.scale_features(X)
        
        return X_scaled, y