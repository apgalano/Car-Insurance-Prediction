#!/usr/bin/env python3
"""
Test script to simulate using the standalone model in an external repository.
This script only uses standard libraries and the pickled model.
"""

import joblib
import pandas as pd

def test_standalone_model():
    """Test the standalone model without any project dependencies."""
    print("🧪 Testing Standalone Model (External Usage)")
    print("=" * 50)
    
    try:
        # Load the standalone model (this should work without source code)
        print("1. Loading standalone model...")
        model = joblib.load('models/car_insurance_standalone.pkl')
        print(f"   ✅ Model loaded: {type(model).__name__}")
        
        # Test single prediction
        print("\n2. Testing single prediction...")
        result = model.predict_single(
            age=30,
            job='management',
            marital='single',
            education='tertiary',
            balance=2000
        )
        
        print(f"   ✅ Prediction: {'YES' if result['will_buy_insurance'] else 'NO'}")
        print(f"   ✅ Confidence: {result['confidence']:.1%}")
        print(f"   ✅ Probability of buying: {result['probability_yes']:.1%}")
        
        # Test batch prediction
        print("\n3. Testing batch prediction...")
        customers = [
            {'age': 25, 'job': 'student', 'balance': 500},
            {'age': 45, 'job': 'management', 'balance': 5000},
            {'age': 60, 'job': 'retired', 'balance': 2000}
        ]
        
        predictions = model.predict(customers)
        probabilities = model.predict_proba(customers)
        
        print("   Batch results:")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            will_buy = "YES" if pred else "NO"
            confidence = max(prob)
            print(f"     Customer {i+1}: {will_buy} (confidence: {confidence:.1%})")
        
        # Test with DataFrame
        print("\n4. Testing with DataFrame...")
        df = pd.DataFrame([{
            'age': 35,
            'job': 'management',
            'marital': 'married',
            'education': 'tertiary',
            'balance': 1500,
            'hh_insurance': 'yes',
            'car_loan': 'no',
            'communication': 'cellular',
            'outcome': 'success',
            'call_start': '09:30:00',
            'call_end': '09:45:00'
        }])
        
        df_prediction = model.predict(df)[0]
        df_proba = model.predict_proba(df)[0]
        
        print(f"   ✅ DataFrame prediction: {'YES' if df_prediction else 'NO'}")
        print(f"   ✅ Confidence: {max(df_proba):.1%}")
        
        print("\n" + "=" * 50)
        print("✅ All tests passed! Standalone model works perfectly!")
        print("\n📋 Usage Summary:")
        print("   • No source code dependencies required")
        print("   • Works with dict, list, or DataFrame input")
        print("   • Provides predictions and confidence scores")
        print("   • Self-contained preprocessing")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False


def show_usage_example():
    """Show a complete usage example for external repositories."""
    print("\n" + "=" * 50)
    print("📖 Usage Example for External Repositories")
    print("=" * 50)
    
    example_code = '''
# External Repository Usage Example
# File: predict_insurance.py

import joblib

# Load the model (only needs joblib, pandas, numpy, sklearn)
model = joblib.load('car_insurance_standalone.pkl')

# Method 1: Single prediction with named parameters
result = model.predict_single(
    age=30,
    job='management',
    marital='married',
    education='tertiary',
    balance=2000,
    hh_insurance='yes'
)

print(f"Will buy insurance: {result['will_buy_insurance']}")
print(f"Confidence: {result['confidence']:.1%}")

# Method 2: Batch prediction with list of dicts
customers = [
    {'age': 25, 'job': 'student', 'balance': 500},
    {'age': 45, 'job': 'management', 'balance': 5000}
]

predictions = model.predict(customers)
probabilities = model.predict_proba(customers)

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f"Customer {i+1}: {'YES' if pred else 'NO'} ({max(prob):.1%})")

# Method 3: DataFrame prediction
import pandas as pd
df = pd.DataFrame(customers)
df_predictions = model.predict(df)
'''
    
    print(example_code)


if __name__ == "__main__":
    success = test_standalone_model()
    
    if success:
        show_usage_example()
    
    exit(0 if success else 1)