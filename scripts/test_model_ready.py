#!/usr/bin/env python3
"""Quick test to verify the model is ready for use."""

from src.models import CarInsurancePredictor, quick_predict

def test_model_availability():
    """Test that the model can be loaded and used."""
    print("🧪 Testing Model Availability")
    print("=" * 35)
    
    try:
        # Test 1: Load predictor
        print("1. Loading predictor...")
        predictor = CarInsurancePredictor()
        print("   ✅ Predictor loaded successfully")
        
        # Test 2: Make a simple prediction
        print("\n2. Testing prediction...")
        result = predictor.predict_customer(
            age=30,
            job='management',
            balance=2000
        )
        print(f"   ✅ Prediction: {'YES' if result['will_buy_insurance'] else 'NO'}")
        print(f"   ✅ Confidence: {result['confidence']:.1%}")
        
        # Test 3: Test quick predict function
        print("\n3. Testing quick predict...")
        will_buy = quick_predict(age=25, job='student', balance=500)
        print(f"   ✅ Quick predict: {'YES' if will_buy else 'NO'}")
        
        # Test 4: Check feature importance
        print("\n4. Testing feature importance...")
        top_features = predictor.get_top_features(5)
        print("   ✅ Top 5 features:")
        for i, (feature, coef) in enumerate(top_features.items(), 1):
            print(f"      {i}. {feature}: {coef:.4f}")
        
        print("\n" + "=" * 35)
        print("✅ All tests passed! Model is ready for use.")
        print("\n📦 Model files available:")
        print("   • models/car_insurance_pipeline.pkl (complete pipeline)")
        print("   • models/car_insurance_portable.py (portable model)")
        print("   • models/README.md (usage guide)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_model_availability()
    exit(0 if success else 1)