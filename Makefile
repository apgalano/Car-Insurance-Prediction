# Car Insurance Prediction Project Makefile

.PHONY: help setup install clean test train demo lint format

# Default target
help:
	@echo "Car Insurance Prediction Project"
	@echo "================================"
	@echo ""
	@echo "Available commands:"
	@echo "  setup     - Set up virtual environment (run ./setup.sh instead)"
	@echo "  install   - Install dependencies (assumes venv is active)
  install-isolated - Install bypassing all pip configuration (including pip.conf)"
	@echo "  demo      - Run project structure demo"
	@echo "  train     - Run the training pipeline"
	@echo "  test      - Run unit tests"
	@echo "  clean     - Clean up generated files"
	@echo "  lint      - Run code linting (requires flake8)"
	@echo "  format    - Format code (requires black)"

# Set up the project (use ./setup.sh instead for full automation)
setup:
	@echo "For automated setup, run: ./setup.sh"
	@echo "For manual setup:"
	python3.11 -m venv venv
	@echo "Activate virtual environment with: source venv/bin/activate"
	@echo "Then run: make install"

# Install dependencies (assumes virtual environment is active)
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .
	@echo "Installation complete!"

# Install bypassing all pip configuration (including pip.conf)
install-isolated:
	pip install --isolated --upgrade pip
	pip install --isolated -r requirements.txt
	pip install --isolated -e .
	@echo "Installation complete bypassing all pip configuration!"

# Run demo
demo:
	python scripts/demo_structure.py

# Run training pipeline
train:
	python scripts/train_model.py

# Run tests
test:
	python -m pytest tests/ -v

# Clean up generated files
clean:
	rm -rf outputs/figures/*
	rm -rf outputs/models/*
	rm -rf outputs/reports/*
	rm -rf src/__pycache__/
	rm -rf tests/__pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete

# Lint code (optional)
lint:
	flake8 src/ tests/ scripts/

# Format code (optional)
format:
	black src/ tests/ scripts/