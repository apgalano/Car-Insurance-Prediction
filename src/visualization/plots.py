"""Visualization utilities for exploratory data analysis."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

from ..config.settings import FIGURES_DIR

sns.set()


class DataVisualizer:
    """Handles data visualization and plotting."""
    
    def __init__(self, output_dir: Path = FIGURES_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_numerical_distributions(self, data: pd.DataFrame) -> None:
        """Plot distributions of numerical features."""
        numerical_cols = ['Age', 'Balance', 'NoOfContacts', 'DaysPassed', 'PrevAttempts', 'CallDuration']
        
        for col in numerical_cols:
            if col in data.columns:
                plt.figure(figsize=(8, 6))
                sns.kdeplot(data[col])
                plt.title(f'Distribution of {col}')
                plt.savefig(self.output_dir / f'{col.lower()}_hist.jpg', bbox_inches='tight')
                plt.close()
    
    def plot_numerical_vs_target(self, data: pd.DataFrame, target_col: str = 'CarInsurance') -> None:
        """Plot numerical features vs target variable."""
        numerical_cols = ['Age', 'Balance', 'NoOfContacts', 'DaysPassed', 'PrevAttempts', 'CallDuration']
        
        for col in numerical_cols:
            if col in data.columns:
                plt.figure(figsize=(10, 6))
                plot = sns.lmplot(x=col, y=target_col, data=data, fit_reg=False, height=6)
                plot.savefig(self.output_dir / f'{col.lower()}_vs_car.jpg', bbox_inches='tight')
                plt.close()
    
    def plot_categorical_distributions(self, data: pd.DataFrame, target_col: str = 'CarInsurance') -> None:
        """Plot distributions of categorical features stacked by target."""
        categorical_cols = [
            'Job', 'Marital', 'Education', 'Default', 'HHInsurance', 'CarLoan',
            'Communication', 'LastContactDay', 'LastContactMonth', 'Outcome'
        ]
        
        # Plot target distribution
        plt.figure(figsize=(8, 6))
        data[target_col].value_counts().plot(kind='bar')
        plt.xlabel('Insurance')
        plt.ylabel('Count')
        plt.title('Target Variable Distribution')
        plt.savefig(self.output_dir / 'insurance.jpg', bbox_inches='tight')
        plt.close()
        
        # Plot categorical features
        for col in categorical_cols:
            if col in data.columns:
                plt.figure(figsize=(12, 6))
                df_grouped = data.groupby(col).apply(lambda x: x[target_col].value_counts(), include_groups=False)
                df_grouped = df_grouped.unstack().fillna(0)
                df_grouped.plot.bar(stacked=True)
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.title(f'{col} Distribution by Target')
                plt.xticks(rotation=45)
                
                # Save with appropriate filename
                filename_map = {
                    'Job': 'job_hist.jpg',
                    'Marital': 'marital_hist.jpg',
                    'Education': 'education_hist.jpg',
                    'Default': 'default_hist.jpg',
                    'HHInsurance': 'house_hist.jpg',
                    'CarLoan': 'loan_hist.jpg',
                    'Communication': 'communication.jpg',
                    'LastContactDay': 'day.jpg',
                    'LastContactMonth': 'month.jpg',
                    'Outcome': 'outcome.jpg'
                }
                
                filename = filename_map.get(col, f'{col.lower()}_hist.jpg')
                plt.savefig(self.output_dir / filename, bbox_inches='tight')
                plt.close()
    
    def generate_all_plots(self, data: pd.DataFrame, target_col: str = 'CarInsurance') -> None:
        """Generate all visualization plots."""
        print("Generating numerical distribution plots...")
        self.plot_numerical_distributions(data)
        
        print("Generating numerical vs target plots...")
        self.plot_numerical_vs_target(data, target_col)
        
        print("Generating categorical distribution plots...")
        self.plot_categorical_distributions(data, target_col)
        
        print(f"All plots saved to {self.output_dir}")