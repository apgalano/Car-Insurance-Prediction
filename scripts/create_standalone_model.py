#!/usr/bin/env python3
"""Create a standalone model that can be used without source code."""

import sys
from pathlib import Path
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.standalone_model import create_standalone_model
from src.config.settings import MODELS_DIR


def main():
    """Create and save the standalone model."""
    print("🔧 Creating Standalone Car Insurance Model")
    print("=" * 45)
    
    try:
        # Create the standalone model
        print("1. Loading existing pipeline...")
        standalone_model = create_standalone_model()
        print("   ✅ Standalone model created")
        
        # Save the standalone model
        print("\n2. Saving standalone model...")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        standalone_path = MODELS_DIR / 'car_insurance_standalone.pkl'
        
        joblib.dump(standalone_model, standalone_path)
        print(f"   ✅ Saved to: {standalone_path}")
        
        # Test the standalone model
        print("\n3. Testing standalone model...")
        
        # Test prediction
        result = standalone_model.predict_single(
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
        
        print(f"   ✅ Test prediction: {'YES' if result['will_buy_insurance'] else 'NO'}")
        print(f"   ✅ Confidence: {result['confidence']:.1%}")
        
        # Show file size
        size_mb = standalone_path.stat().st_size / (1024 * 1024)
        print(f"\n📦 Standalone model size: {size_mb:.2f} MB")
        
        print("\n" + "=" * 45)
        print("✅ Standalone model created successfully!")
        print("\n💡 This model can be used in external repositories with:")
        print("   import joblib")
        print("   model = joblib.load('car_insurance_standalone.pkl')")
        print("   result = model.predict_single(age=30, job='management')")
        
    except Exception as e:
        print(f"\n❌ Failed to create standalone model: {e}")
        raise


if __name__ == "__main__":
    main()