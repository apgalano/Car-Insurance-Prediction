"""Tests for data loader module."""

import unittest
import pandas as pd
from pathlib import Path
import sys

# Add src to path for testing
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = DataLoader()
    
    def test_load_train_data(self):
        """Test loading training data."""
        try:
            train_data = self.loader.load_train_data()
            self.assertIsInstance(train_data, pd.DataFrame)
            self.assertGreater(len(train_data), 0)
        except FileNotFoundError:
            self.skipTest("Training data file not found")
    
    def test_load_test_data(self):
        """Test loading test data."""
        try:
            test_data = self.loader.load_test_data()
            self.assertIsInstance(test_data, pd.DataFrame)
            self.assertGreater(len(test_data), 0)
        except FileNotFoundError:
            self.skipTest("Test data file not found")
    
    def test_get_data_info(self):
        """Test getting data information."""
        try:
            info = self.loader.get_data_info()
            self.assertIn('train_shape', info)
            self.assertIn('test_shape', info)
            self.assertIn('train_columns', info)
            self.assertIn('test_columns', info)
        except FileNotFoundError:
            self.skipTest("Data files not found")


if __name__ == '__main__':
    unittest.main()