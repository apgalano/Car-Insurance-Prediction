"""Configuration settings for the car insurance prediction project."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"  # Changed to be outside outputs/ directory
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Data files
TRAIN_FILE = "carInsurance_train.csv"
TEST_FILE = "carInsurance_test.csv"

# Model parameters
RANDOM_STATE = 1
TEST_SIZE = 0.2
CV_FOLDS = 10

# Outlier thresholds
BALANCE_THRESHOLD = 70000
PREV_ATTEMPTS_THRESHOLD = 35

# Missing value replacements
MISSING_VALUE_REPLACEMENTS = {
    'Job': 'management',
    'Education': 'tertiary',
    'Communication': 'cellular',
    'Outcome': 'other'
}

# Feature columns order
FEATURE_COLUMNS_ORDER = [
    'Age', 'Default', 'Balance', 'HHInsurance', 'CarLoan', 'NoOfContacts',
    'DaysPassed', 'PrevAttempts', 'admin.', 'blue-collar', 'CallDuration',
    'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed',
    'services', 'student', 'technician', 'unemployed', 'divorced', 'married',
    'single', 'primary', 'secondary', 'tertiary', 'cellular', 'telephone',
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 'apr', 'aug', 'dec', 'feb',
    'jan', 'jul', 'jun', 'mar', 'may', 'nov', 'oct', 'sep', 'failure',
    'other', 'success', 'CarInsurance'
]

# Categorical columns for dummy encoding
CATEGORICAL_COLUMNS = [
    'Job', 'Marital', 'Education', 'Communication', 
    'LastContactDay', 'LastContactMonth', 'Outcome'
]