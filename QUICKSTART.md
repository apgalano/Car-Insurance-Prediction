# Quick Start Guide

Get up and running with the Car Insurance Prediction project in minutes!

## Prerequisites

- Python 3.11 installed on your system
- Git (to clone the repository)

## 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd car-insurance-prediction

# Run automated setup (recommended)
./setup.sh
```

The setup script will:
- Create a Python 3.11 virtual environment
- Install all dependencies (bypassing pip.conf)
- Set up the project structure
- Run a demo to verify everything works

## 2. Activate Virtual Environment

Every time you work on the project, activate the virtual environment:

```bash
source venv/bin/activate
```

You'll see `(venv)` in your terminal prompt when it's active.

## 3. Run the Project

```bash
# Run the complete training pipeline
python scripts/train_model.py

# See usage examples
python scripts/example_usage.py

# Or run individual components
python scripts/demo_structure.py
```

## 4. Explore Results

After training, check these directories:
- `outputs/figures/` - Data visualization plots
- `outputs/models/` - Trained model files
- `outputs/reports/` - Analysis reports and feature importance

## 5. Run Tests

```bash
python -m pytest tests/
```

## Alternative: Manual Setup

If you prefer manual setup:

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies (bypassing pip.conf if needed)
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Or to completely bypass pip configuration:
pip install --isolated --upgrade pip
pip install --isolated -r requirements.txt
pip install --isolated -e .
```

## Using Makefile

For convenience, use the provided Makefile:

```bash
# See all available commands
make help

# Set up project
make setup
source venv/bin/activate
make install

# Run training
make train

# Run tests
make test
```

## Troubleshooting

### Python 3.11 Not Found
- **macOS**: `brew install python@3.11`
- **Ubuntu**: `sudo apt install python3.11 python3.11-venv`

### Virtual Environment Issues
```bash
# Check if virtual environment is active
which python  # Should show path to venv/bin/python

# If not active, activate it
source venv/bin/activate
```

### Import Errors
Make sure you've installed the package in development mode:
```bash
pip install -e .
```

## Next Steps

1. **Explore the code**: Check out the modular structure in `src/`
2. **Modify parameters**: Edit `src/config/settings.py`
3. **Add features**: Extend the preprocessing or model classes
4. **Run experiments**: Use Jupyter notebooks in `notebooks/`

Happy coding! 🚀