#!/usr/bin/env python3
"""Main script to train the car insurance prediction model."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.visualization.plots import DataVisualizer
from src.models.logistic_regression import CarInsuranceModel
from src.models.prediction_pipeline import CarInsurancePredictionPipeline
from src.config.settings import REPORTS_DIR


def main():
    """Main training pipeline."""
    print("Starting Car Insurance Prediction Model Training...")
    print("=" * 55)
    
    try:
        # Initialize components
        loader = DataLoader()
        preprocessor = DataPreprocessor()
        visualizer = DataVisualizer()
        model = CarInsuranceModel()
        pipeline = CarInsurancePredictionPipeline()
        
        # Load data
        print("\n1. Loading data...")
        combined_data = loader.load_combined_data()
        print(f"Combined data shape: {combined_data.shape}")
        
        # Generate visualizations
        print("\n2. Generating visualizations...")
        processed_data = preprocessor.create_call_duration_feature(combined_data)
        visualizer.generate_all_plots(processed_data)
        
        # Preprocess data
        print("\n3. Preprocessing data...")
        X, y = preprocessor.preprocess_training_data(combined_data)
        print(f"Processed features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Train model
        print("\n4. Training model...")
        # Get feature names before scaling (X is numpy array after scaling)
        temp_X, temp_y = preprocessor.split_features_target(
            preprocessor.reorder_columns(
                preprocessor.create_dummy_variables(
                    preprocessor.handle_missing_values(
                        preprocessor.remove_outliers(
                            preprocessor.create_call_duration_feature(combined_data).loc[0:3999, :].copy()
                        )
                    )
                )
            )
        )
        feature_names = list(temp_X.columns)
        
        results = model.train(X, y, feature_names=feature_names)
        
        # Print results
        print(f"\n5. Training Results:")
        print(f"Training accuracy: {results['train_accuracy']:.4f}")
        print(f"Testing accuracy: {results['test_accuracy']:.4f}")
        print(f"\nClassification Report:")
        print(results['classification_report'])
        
        # Save feature importance
        print("\n6. Saving results...")
        feature_importance = model.get_feature_importance()
        
        # Create reports directory
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        feature_importance.to_csv(REPORTS_DIR / 'feature_importance.csv', index=False)
        print(f"Feature importance saved to {REPORTS_DIR / 'feature_importance.csv'}")
        
        # Train and save complete pipeline for predictions
        print("\n7. Training and saving prediction pipeline...")
        pipeline_results = pipeline.fit(combined_data)
        pipeline_path = pipeline.save_pipeline()
        print(f"Complete pipeline saved to {pipeline_path}")
        print("   (Contains both model and preprocessor - ready for predictions!)")
        
        # Generate data description
        train_data = loader.load_train_data()
        description = train_data.describe()
        description.to_csv(REPORTS_DIR / 'data_description.csv')
        print(f"Data description saved to {REPORTS_DIR / 'data_description.csv'}")
        
        print("\n" + "=" * 55)
        print("✅ Training completed successfully!")
        print(f"📦 Use the pipeline at: {pipeline_path}")
        print("🔮 Ready for making predictions on new data!")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("\nFor debugging, check:")
        print("1. Data files are in data/raw/ directory")
        print("2. Virtual environment is activated")
        print("3. All dependencies are installed")
        raise


if __name__ == "__main__":
    main()