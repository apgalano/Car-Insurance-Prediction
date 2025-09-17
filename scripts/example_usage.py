#!/usr/bin/env python3
"""Example usage of the car insurance prediction model."""

from src.models import CarInsurancePredictor, quick_predict

def main():
    """Demonstrate different ways to use the model."""
    print("🚗 Car Insurance Prediction - Usage Examples")
    print("=" * 50)
    
    # Method 1: Quick prediction (simplest)
    print("\n1. Quick Prediction (simplest method)")
    print("-" * 40)
    will_buy = quick_predict(age=30, job='management', balance=2000)
    print(f"Customer will buy insurance: {'YES' if will_buy else 'NO'}")
    
    # Method 2: Detailed prediction with confidence
    print("\n2. Detailed Prediction with Confidence")
    print("-" * 40)
    predictor = CarInsurancePredictor()
    
    result = predictor.predict_customer(
        age=35,
        job='management',
        marital='married',
        education='tertiary',
        balance=1500,
        hh_insurance='yes',
        car_loan='no',
        communication='cellular',
        outcome='success',
        call_start='09:30:00',
        call_end='09:45:00'
    )
    
    print(f"Prediction: {'YES' if result['will_buy_insurance'] else 'NO'}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Probability of buying: {result['probability_yes']:.1%}")
    
    # Method 3: Batch predictions
    print("\n3. Batch Predictions")
    print("-" * 40)
    customers = [
        {'age': 25, 'job': 'student', 'balance': 500},
        {'age': 45, 'job': 'management', 'balance': 5000},
        {'age': 60, 'job': 'retired', 'balance': 2000}
    ]
    
    results = predictor.predict_batch(customers)
    for i, (customer, result) in enumerate(zip(customers, results), 1):
        prediction = 'YES' if result['will_buy_insurance'] else 'NO'
        confidence = result['confidence']
        print(f"Customer {i}: {prediction} (confidence: {confidence:.1%})")
    
    # Method 4: Feature importance
    print("\n4. Most Important Features")
    print("-" * 40)
    top_features = predictor.get_top_features(5)
    for i, (feature, importance) in enumerate(top_features.items(), 1):
        direction = "↑" if importance > 0 else "↓"
        print(f"{i}. {feature}: {direction} {abs(importance):.4f}")
    
    print("\n" + "=" * 50)
    print("✅ Ready to use! The model is saved in models/ directory")
    print("📖 See models/README.md for more details")

if __name__ == "__main__":
    main()