"""Data loading utilities."""

import pandas as pd
from pathlib import Path
from typing import Tuple
from ..config.settings import RAW_DATA_DIR, TRAIN_FILE, TEST_FILE


class DataLoader:
    """Handles loading of training and test data."""
    
    def __init__(self, data_dir: Path = RAW_DATA_DIR):
        self.data_dir = data_dir
    
    def load_train_data(self) -> pd.DataFrame:
        """Load training data."""
        train_path = self.data_dir / TRAIN_FILE
        return pd.read_csv(train_path)
    
    def load_test_data(self) -> pd.DataFrame:
        """Load test data."""
        test_path = self.data_dir / TEST_FILE
        return pd.read_csv(test_path)
    
    def load_combined_data(self) -> pd.DataFrame:
        """Load and combine training and test data."""
        train_data = self.load_train_data()
        test_data = self.load_test_data()
        
        combined_data = pd.concat([train_data, test_data])
        combined_data.index = combined_data['Id'] - 1
        
        return combined_data
    
    def get_data_info(self) -> dict:
        """Get basic information about the datasets."""
        train_data = self.load_train_data()
        test_data = self.load_test_data()
        
        return {
            'train_shape': train_data.shape,
            'test_shape': test_data.shape,
            'train_columns': list(train_data.columns),
            'test_columns': list(test_data.columns)
        }