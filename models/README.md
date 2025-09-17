# Trained Models

This directory contains the trained car insurance prediction models that are ready for use.

## Files

- `car_insurance_pipeline.pkl` - Complete prediction pipeline including preprocessor and trained model (this is all you need!)

## Usage

### Quick Prediction
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
    balance=1500,
    # ... other parameters
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

To test if the model is working:
```bash
python scripts/test_model_ready.py
python scripts/example_usage.py
```

This will update the models in this directory with the latest training results.