# Trained Models

This directory contains the trained car insurance prediction models that are ready for use.

## Files

- `car_insurance_pipeline.pkl` - Complete prediction pipeline (requires source code)
- `car_insurance_portable.py` - Portable model (single file, no dependencies) ⭐

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

### Option 2: External Use (Portable Model) ⭐ Recommended
```python
# Copy car_insurance_portable.py to your project
from car_insurance_portable import predict_insurance

# Make a prediction
result = predict_insurance(
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
python scripts/test_truly_portable.py   # Test portable model
```

## Creating Portable Model

To create/update the portable model:
```bash
python scripts/create_portable_model.py
```

## External Repository Usage

For external repositories, use the portable model:

1. **Copy the file**: Copy `models/car_insurance_portable.py` to your project
2. **Install dependencies**: `pip install pandas numpy scikit-learn`
3. **Use the model**:
   ```python
   from car_insurance_portable import predict_insurance
   result = predict_insurance(age=30, job='management')
   ```

## Retraining

To retrain the models with new data:
```bash
python scripts/train_model.py           # Retrain pipeline
python scripts/create_portable_model.py # Update portable version
```

This will update both models in this directory with the latest training results.