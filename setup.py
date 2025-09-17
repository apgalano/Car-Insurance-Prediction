"""Setup script for car insurance prediction package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="car-insurance-prediction",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A logistic regression model for predicting car insurance purchases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/car-insurance-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "train-car-insurance=scripts.train_model:main",
        ],
    },
)