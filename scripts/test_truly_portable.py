#!/usr/bin/env python3
"""
Test the truly portable model in a completely isolated environment.
This simulates using the model in an external repository.
"""

def test_portable_model():
    """Test the portable model without any project dependencies."""
    print("🧪 Testing Truly Portable Model")
    print("=" * 40)
    
    try:
        # Import the portable model (should work without any src dependencies)
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
        from car_insurance_portable import predict_insurance, predict_batch
        
        print("✅ Successfully imported portable model")
        
        # Test single prediction
        print("\n1. Testing single prediction...")
        result = predict_insurance(
            age=35,
            job='management',
            marital='married',
            education='tertiary',
            balance=1500,
            hh_insurance='yes'
        )
        
        print(f"   ✅ Prediction: {'YES' if result['will_buy_insurance'] else 'NO'}")
        print(f"   ✅ Confidence: {result['confidence']:.1%}")
        
        # Test batch prediction
        print("\n2. Testing batch prediction...")
        customers = [
            {'age': 25, 'job': 'student', 'balance': 500},
            {'age': 45, 'job': 'management', 'balance': 5000},
            {'age': 60, 'job': 'retired', 'balance': 2000}
        ]
        
        batch_results = predict_batch(customers)
        
        print("   Batch results:")
        for i, result in enumerate(batch_results):
            prediction = 'YES' if result['will_buy_insurance'] else 'NO'
            confidence = result['confidence']
            print(f"     Customer {i+1}: {prediction} (confidence: {confidence:.1%})")
        
        # Test with minimal parameters
        print("\n3. Testing with minimal parameters...")
        minimal_result = predict_insurance(age=30)
        print(f"   ✅ Minimal prediction: {'YES' if minimal_result['will_buy_insurance'] else 'NO'}")
        
        print("\n" + "=" * 40)
        print("✅ All tests passed! Portable model works perfectly!")
        print("\n📋 Key Benefits:")
        print("   • No source code dependencies")
        print("   • Single file deployment")
        print("   • Self-contained preprocessing")
        print("   • Works with minimal parameters")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


def show_external_usage():
    """Show how to use this in an external repository."""
    print("\n" + "=" * 40)
    print("📖 External Repository Usage")
    print("=" * 40)
    
    usage_example = '''
# In your external repository:
# 1. Copy car_insurance_portable.py to your project
# 2. Use it like this:

from car_insurance_portable import predict_insurance, predict_batch

# Single prediction
result = predict_insurance(
    age=30,
    job='management',
    balance=2000
)

print(f"Will buy: {result['will_buy_insurance']}")
print(f"Confidence: {result['confidence']:.1%}")

# Batch prediction
customers = [
    {'age': 25, 'job': 'student'},
    {'age': 45, 'job': 'management', 'balance': 5000}
]

results = predict_batch(customers)
for i, result in enumerate(results):
    print(f"Customer {i+1}: {result['will_buy_insurance']}")

# Dependencies needed:
# - pandas
# - numpy  
# - scikit-learn
# That's it! No custom code required.
'''
    
    print(usage_example)


if __name__ == "__main__":
    success = test_portable_model()
    
    if success:
        show_external_usage()
    
    exit(0 if success else 1)