"""Models module."""

from .logistic_regression import CarInsuranceModel
from .prediction_pipeline import CarInsurancePredictionPipeline
from .predictor import CarInsurancePredictor, quick_predict

__all__ = [
    'CarInsuranceModel',
    'CarInsurancePredictionPipeline', 
    'CarInsurancePredictor',
    'quick_predict'
]