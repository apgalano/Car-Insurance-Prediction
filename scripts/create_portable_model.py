#!/usr/bin/env python3
"""
Create a truly portable model that can be loaded without any source dependencies.
This script creates a standalone Python file with the model embedded.
"""

import sys
from pathlib import Path
import joblib
import pickle
import base64

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.models.prediction_pipeline import CarInsurancePredictionPipeline
from src.config.settings import MODELS_DIR


def create_portable_model():
    """Create a portable model file that can be used anywhere."""
    
    # Load the existing pipeline
    pipeline = CarInsurancePredictionPipeline()
    pipeline.load_pipeline(MODELS_DIR / 'car_insurance_pipeline.pkl')
    
    # Extract the components we need
    model = pipeline.model.model
    scaler = pipeline.preprocessor.scaler
    feature_names = pipeline.feature_names
    
    # Create the portable model code
    portable_code = f'''#!/usr/bin/env python3
"""
Portable Car Insurance Prediction Model
Generated automatically - no external dependencies required.

Usage:
    from car_insurance_portable import predict_insurance
    
    result = predict_insurance(
        age=30,
        job='management',
        balance=2000
    )
    print(f"Will buy: {{result['will_buy_insurance']}}")
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import base64

# Embedded model components (base64 encoded)
MODEL_DATA = """{base64.b64encode(pickle.dumps(model)).decode()}"""
SCALER_DATA = """{base64.b64encode(pickle.dumps(scaler)).decode()}"""
FEATURE_NAMES = {feature_names}

# Configuration constants
MISSING_VALUE_REPLACEMENTS = {{
    'Job': 'management',
    'Education': 'tertiary', 
    'Communication': 'cellular',
    'Outcome': 'other'
}}

CATEGORICAL_COLUMNS = [
    'Job', 'Marital', 'Education', 'Communication',
    'LastContactDay', 'LastContactMonth', 'Outcome'
]

BALANCE_THRESHOLD = 70000
PREV_ATTEMPTS_THRESHOLD = 35

# Load embedded components
_model = pickle.loads(base64.b64decode(MODEL_DATA))
_scaler = pickle.loads(base64.b64decode(SCALER_DATA))


def _create_call_duration_feature(data):
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


def _handle_missing_values(data):
    """Handle missing values using predefined replacements."""
    data = data.copy()
    
    for column, replacement in MISSING_VALUE_REPLACEMENTS.items():
        if column in data.columns:
            data[column] = data[column].fillna(replacement)
    
    return data


def _create_dummy_variables(data):
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


def _preprocess_data(data):
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
    data = _create_call_duration_feature(data)
    data = _handle_missing_values(data)
    data = _create_dummy_variables(data)
    
    # Convert all column names to strings
    data.columns = data.columns.astype(str)
    
    # Ensure all training features are present (add missing dummy columns)
    for feature in FEATURE_NAMES:
        if feature not in data.columns:
            data[feature] = 0
    
    # Select only the features used in training (in correct order)
    data = data[FEATURE_NAMES]
    
    # Convert all columns to numeric (in case any string values remain)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
    
    return data


def predict_insurance(age, job='unknown', marital='unknown', education='unknown',
                     default='no', balance=0, hh_insurance='no', car_loan='no',
                     no_of_contacts=0, days_passed=0, prev_attempts=0,
                     communication='cellular', last_contact_day=1,
                     last_contact_month='jan', outcome='other',
                     call_start='09:00:00', call_end='09:05:00'):
    """
    Predict if a customer will buy car insurance.
    
    Args:
        age: Customer age
        job: Job category
        marital: Marital status
        education: Education level
        default: Has credit in default?
        balance: Average yearly balance
        hh_insurance: Has house insurance?
        car_loan: Has car loan?
        no_of_contacts: Number of contacts during campaign
        days_passed: Days since last contact
        prev_attempts: Number of previous attempts
        communication: Contact communication type
        last_contact_day: Last contact day of month
        last_contact_month: Last contact month
        outcome: Outcome of previous campaign
        call_start: Call start time
        call_end: Call end time
    
    Returns:
        Dictionary with prediction results
    """
    
    # Create input data
    customer_data = pd.DataFrame([{{
        'Age': age,
        'Job': job,
        'Marital': marital,
        'Education': education,
        'Default': default,
        'Balance': balance,
        'HHInsurance': hh_insurance,
        'CarLoan': car_loan,
        'NoOfContacts': no_of_contacts,
        'DaysPassed': days_passed,
        'PrevAttempts': prev_attempts,
        'Communication': communication,
        'LastContactDay': last_contact_day,
        'LastContactMonth': last_contact_month,
        'Outcome': outcome,
        'CallStart': call_start,
        'CallEnd': call_end
    }}])
    
    # Preprocess the data
    processed_data = _preprocess_data(customer_data)
    
    # Scale the features
    scaled_data = _scaler.transform(processed_data)
    
    # Make prediction
    prediction = _model.predict(scaled_data)[0]
    probabilities = _model.predict_proba(scaled_data)[0]
    
    return {{
        'prediction': int(prediction),
        'will_buy_insurance': bool(prediction),
        'probability_no': float(probabilities[0]),
        'probability_yes': float(probabilities[1]),
        'confidence': float(max(probabilities))
    }}


def predict_batch(customers):
    """
    Predict for multiple customers.
    
    Args:
        customers: List of customer dictionaries or DataFrame
        
    Returns:
        List of prediction results
    """
    if isinstance(customers, list):
        customers = pd.DataFrame(customers)
    
    # Preprocess the data
    processed_data = _preprocess_data(customers)
    
    # Scale the features
    scaled_data = _scaler.transform(processed_data)
    
    # Make predictions
    predictions = _model.predict(scaled_data)
    probabilities = _model.predict_proba(scaled_data)
    
    results = []
    for pred, prob in zip(predictions, probabilities):
        results.append({{
            'prediction': int(pred),
            'will_buy_insurance': bool(pred),
            'probability_no': float(prob[0]),
            'probability_yes': float(prob[1]),
            'confidence': float(max(prob))
        }})
    
    return results


# Example usage
if __name__ == "__main__":
    # Test the model
    result = predict_insurance(
        age=35,
        job='management',
        marital='married',
        education='tertiary',
        balance=1500
    )
    
    print(f"Will buy insurance: {{result['will_buy_insurance']}}")
    print(f"Confidence: {{result['confidence']:.1%}}")
'''
    
    return portable_code


def main():
    """Create the portable model file."""
    print("🔧 Creating Portable Car Insurance Model")
    print("=" * 45)
    
    try:
        print("1. Loading existing pipeline...")
        portable_code = create_portable_model()
        
        print("2. Creating portable model file...")
        with open('car_insurance_portable.py', 'w') as f:
            f.write(portable_code)
        
        print("   ✅ Created: car_insurance_portable.py")
        
        # Test the portable model
        print("3. Testing portable model...")
        import subprocess
        result = subprocess.run([
            sys.executable, '-c',
            '''
import sys
sys.path.insert(0, ".")
from car_insurance_portable import predict_insurance

result = predict_insurance(age=30, job="management", balance=2000)
print(f"Test result: {result['will_buy_insurance']} (confidence: {result['confidence']:.1%})")
'''
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"   ✅ {result.stdout.strip()}")
        else:
            print(f"   ❌ Test failed: {result.stderr}")
        
        # Show file size
        file_size = Path('car_insurance_portable.py').stat().st_size / 1024
        print(f"📦 Portable model size: {file_size:.1f} KB")
        
        print("\n" + "=" * 45)
        print("✅ Portable model created successfully!")
        print("\n💡 Usage in external repositories:")
        print("   1. Copy car_insurance_portable.py to your project")
        print("   2. from car_insurance_portable import predict_insurance")
        print("   3. result = predict_insurance(age=30, job='management')")
        
    except Exception as e:
        print(f"\n❌ Failed to create portable model: {e}")
        raise


if __name__ == "__main__":
    main()