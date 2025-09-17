#!/usr/bin/env python3
"""Script to demonstrate making predictions with the trained pipeline."""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.prediction_pipeline import CarInsurancePredictionPipeline
from src.config.settings import MODELS_DIR


def load_pipeline():
    """Load the trained prediction pipeline."""
    pipeline_path = MODELS_DIR / 'car_insurance_pipeline.pkl'
    
    if not pipeline_path.exists():
        print(f"❌ Pipeline not found at {pipeline_path}")
        print("Please run 'python scripts/train_model.py' first to train the model.")
        return None
    
    pipeline = CarInsurancePredictionPipeline()
    pipeline.load_pipeline(pipeline_path)
    print(f"✅ Pipeline loaded from {pipeline_path}")
    return pipeline


def demo_single_prediction(pipeline):
    """Demonstrate single customer prediction."""
    print("\n🔮 Single Customer Prediction Demo")
    print("-" * 40)
    
    # Example customer data
    customer = {
        'Age': 35,
        'Job': 'management',
        'Marital': 'married',
        'Education': 'tertiary',
        'Default': 'no',
        'Balance': 1500,
        'HHInsurance': 'yes',
        'CarLoan': 'no',
        'NoOfContacts': 2,
        'DaysPassed': 180,
        'PrevAttempts': 1,
        'Communication': 'cellular',
        'LastContactDay': 15,
        'LastContactMonth': 'may',
        'Outcome': 'success',
        'CallStart': '09:30:00',
        'CallEnd': '09:45:00'
    }
    
    print("Customer Profile:")
    for key, value in customer.items():
        print(f"  {key}: {value}")
    
    # Make prediction
    result = pipeline.predict_single(**customer)
    
    print(f"\nPrediction Results:")
    print(f"  Will buy insurance: {'YES' if result['will_buy_insurance'] else 'NO'}")
    print(f"  Confidence: {result['confidence']:.1%}")
    print(f"  Probability of buying: {result['probability_yes']:.1%}")
    print(f"  Probability of not buying: {result['probability_no']:.1%}")


def demo_batch_predictions(pipeline):
    """Demonstrate batch predictions."""
    print("\n📊 Batch Predictions Demo")
    print("-" * 40)
    
    # Example batch of customers
    customers = [
        {
            'Age': 25, 'Job': 'student', 'Marital': 'single', 'Education': 'secondary',
            'Default': 'no', 'Balance': 500, 'HHInsurance': 'no', 'CarLoan': 'no',
            'NoOfContacts': 1, 'DaysPassed': 30, 'PrevAttempts': 0,
            'Communication': 'cellular', 'LastContactDay': 10, 'LastContactMonth': 'jun',
            'Outcome': 'other', 'CallStart': '14:00:00', 'CallEnd': '14:05:00'
        },
        {
            'Age': 45, 'Job': 'management', 'Marital': 'married', 'Education': 'tertiary',
            'Default': 'no', 'Balance': 5000, 'HHInsurance': 'yes', 'CarLoan': 'yes',
            'NoOfContacts': 3, 'DaysPassed': 90, 'PrevAttempts': 2,
            'Communication': 'cellular', 'LastContactDay': 20, 'LastContactMonth': 'oct',
            'Outcome': 'success', 'CallStart': '10:00:00', 'CallEnd': '10:20:00'
        },
        {
            'Age': 60, 'Job': 'retired', 'Marital': 'divorced', 'Education': 'primary',
            'Default': 'no', 'Balance': 2000, 'HHInsurance': 'no', 'CarLoan': 'no',
            'NoOfContacts': 1, 'DaysPassed': 365, 'PrevAttempts': 0,
            'Communication': 'telephone', 'LastContactDay': 5, 'LastContactMonth': 'dec',
            'Outcome': 'failure', 'CallStart': '16:00:00', 'CallEnd': '16:03:00'
        }
    ]
    
    # Make batch predictions
    predictions = pipeline.predict(customers)
    probabilities = pipeline.predict_proba(customers)
    
    print(f"Predictions for {len(customers)} customers:")
    for i, (customer, pred, prob) in enumerate(zip(customers, predictions, probabilities)):
        will_buy = "YES" if pred else "NO"
        confidence = max(prob)
        print(f"  Customer {i+1}: {will_buy} (confidence: {confidence:.1%})")


def demo_feature_importance(pipeline):
    """Show feature importance."""
    print("\n📈 Feature Importance")
    print("-" * 40)
    
    importance = pipeline.get_feature_importance()
    
    print("Top 10 most important features:")
    for i, row in importance.head(10).iterrows():
        feature = row['Feature']
        coef = row['Coefficient']
        direction = "↑" if coef > 0 else "↓"
        print(f"  {i+1:2d}. {feature:<20} {direction} {abs(coef):.4f}")


def main():
    """Main demonstration function."""
    print("🚀 Car Insurance Prediction Demo")
    print("=" * 50)
    
    # Load the trained pipeline
    pipeline = load_pipeline()
    if pipeline is None:
        return
    
    try:
        # Run demonstrations
        demo_single_prediction(pipeline)
        demo_batch_predictions(pipeline)
        demo_feature_importance(pipeline)
        
        print("\n" + "=" * 50)
        print("✅ Prediction demo completed successfully!")
        print("\n💡 Usage Tips:")
        print("• Use predict_single(**kwargs) for individual predictions")
        print("• Use predict(dataframe) for batch predictions")
        print("• Use predict_proba() to get probability scores")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()