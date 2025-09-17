# Setup Options

Choose the setup method that works best for you:

## 🚀 Option 1: Automated Setup (Recommended)

```bash
./setup.sh
source venv/bin/activate
```

## 🔧 Option 2: Using Makefile

```bash
make setup
source venv/bin/activate
make install
make demo  # Test the setup
```

## 📋 Option 3: Manual Setup

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Test the setup
python scripts/demo_structure.py
```

## 🐍 Option 4: Using pyenv (Advanced)

If you use pyenv for Python version management:

```bash
# Install Python 3.11 via pyenv
pyenv install 3.11.6
pyenv local 3.11.6

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 📦 Option 5: Using conda/mamba

```bash
# Create conda environment
conda create -n car-insurance python=3.11
conda activate car-insurance

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## ✅ Verify Your Setup

After any setup method, verify everything works:

```bash
# Check Python version
python --version  # Should show Python 3.11.x

# Check virtual environment
which python  # Should point to venv/bin/python

# Run demo
python scripts/demo_structure.py

# Run tests
python -m pytest tests/
```

## 🔄 Daily Workflow

Once set up, your daily workflow is simple:

```bash
# Activate virtual environment
source venv/bin/activate

# Work on the project
python scripts/train_model.py

# Deactivate when done
deactivate
```

## 🆘 Troubleshooting

### Python 3.11 Not Available
- **macOS**: `brew install python@3.11`
- **Ubuntu/Debian**: `sudo apt install python3.11 python3.11-venv`
- **CentOS/RHEL**: `sudo yum install python3.11`


### Permission Issues
```bash
chmod +x setup.sh
./setup.sh
```

### Virtual Environment Not Activating
```bash
# Make sure you're in the project directory
cd car-insurance-prediction

# Try absolute path
source $(pwd)/venv/bin/activate
```

Choose the option that fits your workflow and system setup!