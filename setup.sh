#!/bin/bash

# Car Insurance Prediction Project Setup Script
# This script sets up the complete development environment

set -e  # Exit on any error

echo "🚀 Setting up Car Insurance Prediction Project"
echo "=============================================="

# Use the current Python version
PYTHON_CMD="python"

echo "✅ Using Python: $($PYTHON_CMD --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
$PYTHON_CMD -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
python -m pip install --isolated --upgrade pip

# Install dependencies one by one bypassing all pip configuration
echo "📚 Installing core dependencies..."
python -m pip install --isolated pandas
python -m pip install --isolated numpy
python -m pip install --isolated scikit-learn
python -m pip install --isolated matplotlib
python -m pip install --isolated seaborn
python -m pip install --isolated joblib
python -m pip install --isolated pytest

# Install package in development mode
echo "🔨 Installing package in development mode..."
python -m pip install --isolated -e .

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/raw data/processed outputs/figures outputs/models outputs/reports

# Run demo to verify setup
echo "🧪 Testing installation..."
python scripts/demo_structure.py

echo ""
echo "✅ Setup complete! 🎉"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run training pipeline: python scripts/train_model.py"
echo "3. Make predictions: python scripts/example_usage.py"
echo ""
echo "For help: python scripts/demo_structure.py"