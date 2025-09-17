# Car Insurance Prediction

A machine learning project that predicts whether a customer is likely to purchase car insurance using logistic regression.

## Project Overview

This project uses a dataset from [Kaggle](https://www.kaggle.com/kondla/carinsurance) to build a predictive model for car insurance purchases. The solution includes comprehensive data analysis, visualization, preprocessing, and model training with proper code organization and testing.

## Project Structure

```
car-insurance-prediction/
├── src/                          # Source code
│   ├── config/                   # Configuration settings
│   ├── data/                     # Data loading and preprocessing
│   ├── models/                   # Machine learning models
│   ├── visualization/            # Data visualization
│   └── utils/                    # Utility functions
├── data/                         # Data files
│   ├── raw/                      # Original datasets
│   └── processed/                # Processed datasets
├── outputs/                      # Generated outputs
│   ├── figures/                  # Visualization plots
│   ├── models/                   # Trained models
│   └── reports/                  # Analysis reports
├── tests/                        # Unit tests
├── scripts/                      # Executable scripts
│   ├── train_model.py            # Train the model
│   ├── example_usage.py          # Usage examples
│   ├── make_predictions.py       # Prediction demos
│   ├── test_model_ready.py       # Test model availability
│   ├── check_outputs.py          # Check training outputs
│   └── demo_structure.py         # Project structure demo
└── notebooks/                    # Jupyter notebooks
```

## Prerequisites

- Python 3.11 or higher
- pip (Python package installer)

> 🚀 **New to the project?** Check out our [Quick Start Guide](QUICKSTART.md) for the fastest way to get running!

## Installation

### Option 1: Automated Setup (Recommended)

```bash
git clone <repository-url>
cd car-insurance-prediction
./setup.sh
source venv/bin/activate
```

### Option 2: Manual Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd car-insurance-prediction
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment with Python 3.11
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

3. Upgrade pip and install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt

# If you need to bypass all pip configuration (including pip.conf):
pip install --isolated -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

5. Verify installation:
```bash
python scripts/demo_structure.py
```

## Usage

### Quick Start

Make sure your virtual environment is activated, then run the complete training pipeline:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Run the training pipeline
python scripts/train_model.py

# Test the model and see examples
python scripts/example_usage.py
```

### Making Predictions (Models Ready!)

The repository includes a pre-trained model pipeline in the `models/` directory:

```python
from src.models import CarInsurancePredictor, quick_predict

# Quick prediction
will_buy = quick_predict(age=30, job='management', balance=2000)

# Detailed prediction
predictor = CarInsurancePredictor()
result = predictor.predict_customer(
    age=35,
    job='management',
    marital='married',
    education='tertiary',
    balance=1500
)
print(f"Will buy: {result['will_buy_insurance']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### Training New Models

```python
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.logistic_regression import CarInsuranceModel

# Load data
loader = DataLoader()
data = loader.load_combined_data()

# Preprocess data
preprocessor = DataPreprocessor()
X, y = preprocessor.preprocess_training_data(data)

# Train model
model = CarInsuranceModel()
results = model.train(X, y)
```

## Key Features

- **Modular Design**: Clean separation of concerns with dedicated modules
- **Configuration Management**: Centralized settings and parameters
- **Comprehensive Preprocessing**: Feature engineering, outlier removal, missing value handling
- **Rich Visualizations**: Automated generation of exploratory data analysis plots
- **Ready-to-Use Models**: Pre-trained models available in `models/` directory
- **Easy Prediction API**: Simple interfaces for making predictions
- **Model Persistence**: Save and load trained models
- **Testing**: Unit tests for core functionality
- **Documentation**: Well-documented code with type hints

## Model Performance

The logistic regression model achieves:
- Training accuracy: ~82.8%
- Cross-validation with 10 folds to prevent overfitting
- Feature importance analysis to identify key predictors

## Key Insights

- **Positive predictors**: Call duration, successful previous campaigns, specific months
- **Negative predictors**: Existing house/car loans, high contact frequency
- **Surprising finding**: Account balance has minimal predictive power

## Available Scripts

The `scripts/` directory contains various utility scripts:

- **`train_model.py`** - Complete training pipeline
- **`example_usage.py`** - Usage examples and demonstrations
- **`make_predictions.py`** - Interactive prediction demos
- **`test_model_ready.py`** - Test if model is ready for use
- **`check_outputs.py`** - Verify training outputs
- **`demo_structure.py`** - Show project structure

```bash
# Run any script
python scripts/<script_name>.py
```

## Development

### Running Tests
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Install pytest if not already installed
pip install pytest

# Run tests
python -m pytest tests/
```

### Adding New Features
1. Follow the existing module structure
2. Add configuration parameters to `src/config/settings.py`
3. Write unit tests for new functionality
4. Update documentation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.
