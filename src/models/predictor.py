"""Simple predictor interface for car insurance model."""

from pathlib import Path
from typing import Dict, Any, Union, List
import warnings

from .prediction_pipeline import CarInsurancePredictionPipeline
from ..config.settings import MODELS_DIR


class CarInsurancePredictor:
    """Simple interface for making car insurance predictions."""
    
    def __init__(self, model_path: Path = None):
        """Initialize the predictor with a trained model."""
        self.pipeline = CarInsurancePredictionPipeline()
        
        if model_path is None:
            model_path = MODELS_DIR / 'car_insurance_pipeline.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please train the model first using 'python scripts/train_model.py'"
            )
        
        self.pipeline.load_pipeline(model_path)
        print(f"✅ Car insurance predictor loaded from {model_path}")
    
    def predict_customer(self, 
                        age: int,
                        job: str = 'unknown',
                        marital: str = 'unknown',
                        education: str = 'unknown',
                        default: str = 'no',
                        balance: float = 0,
                        hh_insurance: str = 'no',
                        car_loan: str = 'no',
                        no_of_contacts: int = 0,
                        days_passed: int = 0,
                        prev_attempts: int = 0,
                        communication: str = 'cellular',
                        last_contact_day: int = 1,
                        last_contact_month: str = 'jan',
                        outcome: str = 'other',
                        call_start: str = '09:00:00',
                        call_end: str = '09:05:00') -> Dict[str, Any]:
        """
        Predict if a customer will buy car insurance.
        
        Args:
            age: Customer age
            job: Job category (admin, blue-collar, entrepreneur, housemaid, management, 
                 retired, self-employed, services, student, technician, unemployed)
            marital: Marital status (divorced, married, single)
            education: Education level (primary, secondary, tertiary)
            default: Has credit in default? (yes, no)
            balance: Average yearly balance in euros
            hh_insurance: Has house insurance? (yes, no)
            car_loan: Has car loan? (yes, no)
            no_of_contacts: Number of contacts during campaign
            days_passed: Days since last contact from previous campaign
            prev_attempts: Number of contacts before this campaign
            communication: Contact communication type (cellular, telephone)
            last_contact_day: Last contact day of month
            last_contact_month: Last contact month (jan, feb, mar, apr, may, jun,
                               jul, aug, sep, oct, nov, dec)
            outcome: Outcome of previous campaign (failure, other, success)
            call_start: Call start time (HH:MM:SS)
            call_end: Call end time (HH:MM:SS)
        
        Returns:
            Dictionary with prediction results
        """
        
        customer_data = {
            'Age': age,
            'Job': job,
            'Marital': marital,
            'Education': education,
            'Default': default,
            'Balance': balance,
            'HHInsurance': hh_insurance,
            'CarLoan': car_loan,
            'NoOfContacts': no_of_contacts,
            'DaysPassed': days_passed,
            'PrevAttempts': prev_attempts,
            'Communication': communication,
            'LastContactDay': last_contact_day,
            'LastContactMonth': last_contact_month,
            'Outcome': outcome,
            'CallStart': call_start,
            'CallEnd': call_end
        }
        
        return self.pipeline.predict_single(**customer_data)
    
    def predict_batch(self, customers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict for multiple customers.
        
        Args:
            customers: List of customer dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        for customer in customers:
            try:
                result = self.pipeline.predict_single(**customer)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'prediction': None,
                    'will_buy_insurance': None,
                    'probability_no': None,
                    'probability_yes': None,
                    'confidence': None
                })
        
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance as a dictionary."""
        importance_df = self.pipeline.get_feature_importance()
        
        # Convert to dictionary, excluding intercept
        importance_dict = {}
        for _, row in importance_df.iterrows():
            if row['Feature'] != 'Intercept':
                importance_dict[row['Feature']] = float(row['Coefficient'])
        
        return importance_dict
    
    def get_top_features(self, n: int = 10) -> Dict[str, float]:
        """Get top N most important features."""
        all_features = self.get_feature_importance()
        
        # Sort by absolute coefficient value
        sorted_features = sorted(all_features.items(), 
                               key=lambda x: abs(x[1]), 
                               reverse=True)
        
        return dict(sorted_features[:n])


# Convenience function for quick predictions
def quick_predict(age: int, **kwargs) -> bool:
    """
    Quick prediction function that returns True/False.
    
    Args:
        age: Customer age (required)
        **kwargs: Other customer attributes
        
    Returns:
        True if customer will likely buy insurance, False otherwise
    """
    try:
        predictor = CarInsurancePredictor()
        result = predictor.predict_customer(age=age, **kwargs)
        return result['will_buy_insurance']
    except Exception as e:
        warnings.warn(f"Prediction failed: {e}")
        return False