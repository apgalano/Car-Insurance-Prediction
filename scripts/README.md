# Scripts Directory

This directory contains executable scripts for various project tasks.

## Available Scripts

### Core Scripts
- **`train_model.py`** - Complete model training pipeline
  ```bash
  python scripts/train_model.py
  ```

- **`example_usage.py`** - Usage examples and demonstrations
  ```bash
  python scripts/example_usage.py
  ```

### Demo and Testing Scripts
- **`make_predictions.py`** - Interactive prediction demonstrations
  ```bash
  python scripts/make_predictions.py
  ```

- **`test_model_ready.py`** - Test if the trained model is ready for use
  ```bash
  python scripts/test_model_ready.py
  ```

- **`check_outputs.py`** - Verify training outputs and generated files
  ```bash
  python scripts/check_outputs.py
  ```

- **`demo_structure.py`** - Show project structure and setup instructions
  ```bash
  python scripts/demo_structure.py
  ```

## Usage

All scripts should be run from the project root directory with the virtual environment activated:

```bash
# Activate virtual environment
source venv/bin/activate

# Run any script
python scripts/<script_name>.py
```

## Script Dependencies

All scripts automatically handle Python path setup and can access the `src/` modules without additional configuration.