# Trained Models

This directory contains the trained car insurance prediction models that are ready for use.

## Files

- `car_insurance_pipeline.pkl` - Complete prediction pipeline (requires source code)
- `car_insurance_standalone.pkl` - Standalone model for external use (no source code needed)

## Usage

### Option 1: With Source Code (Full Pipeline)
```python
from src.models import CarInsurancePredictor

# Load the predictor
predictor = CarInsurancePredictor()

# Make a prediction
result = predictor.predict_customer(
    age=35,
    job='management',
    marital='married',
    education='tertiary',
    balance=1500
)

print(f"Will buy insurance: {result['will_buy_insurance']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### Option 2: External Use (Standalone Model)
```python
import joblib

# Load the standalone model (no source code needed)
model = joblib.load('car_insurance_standalone.pkl')

# Make a prediction
result = model.predict_single(
    age=35,
    job='management',
    marital='married',
    education='tertiary',
    balance=1500
)

print(f"Will buy insurance: {result['will_buy_insurance']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### Batch Predictions
```python
from src.models import CarInsurancePredictionPipeline
import pandas as pd

# Load pipeline
pipeline = CarInsurancePredictionPipeline()
pipeline.load_pipeline('models/car_insurance_pipeline.pkl')

# Make predictions on new data
new_customers = pd.DataFrame([...])  # Your customer data
predictions = pipeline.predict(new_customers)
```

## Model Performance

- Training Accuracy: ~82%
- Testing Accuracy: ~84%
- Cross-validation: 10-fold CV
- Features: 74 engineered features

## Retraining

To retrain the models with new data:
```bash
python scripts/train_model.py
```

## Testing

To test if the models are working:
```bash
python scripts/test_model_ready.py      # Test pipeline model
python scripts/example_usage.py         # Usage examples
python test_standalone_external.py      # Test standalone model
```

## Creating Standalone Model

To create/update the standalone model:
```bash
python scripts/create_standalone_model.py
```

## Retraining

To retrain the models with new data:
```bash
python scripts/train_model.py
python scripts/create_standalone_model.py  # Update standalone version
```

This will update both models in this directory with the latest training results.