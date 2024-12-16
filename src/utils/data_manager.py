"""
Data manager singleton for handling stock data and analysis
"""

from typing import Dict, Optional, Tuple, Union
import pandas as pd
from datetime import datetime
from ..analysis.data_processor import DataProcessor
from ..analysis.forecast import Forecaster
from . import get_logger, get_config

logger = get_logger()
config = get_config()

class DataManager:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize data manager with processors"""
        if not self._initialized:
            self._data_processor = DataProcessor()
            self._forecaster = Forecaster()
            self._current_data = None
            self._current_symbol = None
            self._initialized = True
        
    @classmethod
    def get_instance(cls) -> 'DataManager':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = DataManager()
        return cls._instance
        
    def cleanup(self):
        """Clean up resources"""
        try:
            if self._data_processor:
                self._data_processor.cleanup()
            if self._forecaster:
                self._forecaster.cleanup()
            self._current_data = None
            self._current_symbol = None
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            
    @property
    def current_symbol(self) -> Optional[str]:
        """Get current symbol being analyzed"""
        return self._current_symbol
        
    @property
    def current_data(self) -> Optional[pd.DataFrame]:
        """Get current data being analyzed"""
        return self._current_data
        
    def fetch_data(self, symbol: str, start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Fetch and process stock data"""
        try:
            # Validate inputs
            if not symbol or not isinstance(symbol, str):
                raise ValueError("Invalid symbol provided")
                
            # Fetch raw data
            data = self._data_processor.fetch_data(symbol, start_date, end_date)
            
            # Handle single vs multiple ticker response
            if isinstance(data, dict):
                # Store current data as the first ticker's data
                if data:
                    first_symbol = next(iter(data))
                    self._current_data = data[first_symbol]
                    self._current_symbol = first_symbol
            else:
                # Single ticker case
                self._current_data = data
                self._current_symbol = symbol.split(',')[0].strip().upper()
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
            
    def get_technical_indicators(self, data: Optional[pd.DataFrame] = None,
                               selected_indicators: Optional[Dict[str, bool]] = None) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            # Use provided data or current data
            df = data if data is not None else self._current_data
            if df is None:
                raise ValueError("No data available for analysis")
                
            return self._data_processor.calculate_indicators(df, selected_indicators)
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise
            
    def generate_forecast(self, data: Optional[pd.DataFrame] = None, days: int = 30,
                         confidence: float = 0.95) -> Tuple[pd.DataFrame, Dict]:
        """Generate price forecast"""
        try:
            # Use provided data or current data
            df = data if data is not None else self._current_data
            if df is None:
                raise ValueError("No data available for forecasting")
                
            return self._forecaster.generate_forecast(df, days, confidence)
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise
            
    def get_summary_stats(self, data: Optional[pd.DataFrame] = None) -> Dict:
        """Get summary statistics"""
        try:
            # Use provided data or current data
            df = data if data is not None else self._current_data
            if df is None:
                raise ValueError("No data available for analysis")
                
            stats = {
                'data_quality': {
                    'total_rows': len(df),
                    'missing_values': df.isnull().sum().sum(),
                    'date_range': f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
                }
            }
            
            # Add trend analysis if available
            if self._forecaster:
                forecast_data, _ = self.generate_forecast(df, days=1)
                if not forecast_data.empty:
                    trend = forecast_data['trend'].diff().mean()
                    stats['trend_analysis'] = {
                        'direction': 'Upward' if trend > 0 else 'Downward' if trend < 0 else 'Sideways',
                        'strength': abs(trend)
                    }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting summary stats: {str(e)}")
            return {}
            
    def remove_outliers(self, data: Optional[pd.DataFrame] = None, n_sigmas: float = 3) -> pd.DataFrame:
        """Remove outliers from data"""
        try:
            # Use provided data or current data
            df = data if data is not None else self._current_data
            if df is None:
                raise ValueError("No data available for analysis")
                
            return self._data_processor._remove_outliers(df, n_sigmas)
        except Exception as e:
            logger.error(f"Error removing outliers: {str(e)}")
            raise
            
    def validate_forecast(self, data: Optional[pd.DataFrame] = None,
                         forecast: Optional[pd.DataFrame] = None) -> float:
        """Validate forecast accuracy"""
        try:
            # Use provided data or current data
            df = data if data is not None else self._current_data
            if df is None:
                raise ValueError("No data available for validation")
                
            if forecast is None:
                forecast, _ = self.generate_forecast(df)
                
            return self._forecaster.validate_forecast(df, forecast)
        except Exception as e:
            logger.error(f"Error validating forecast: {str(e)}")
            return 0.0
            
    def analyze_seasonality(self, data: Optional[pd.DataFrame] = None) -> Dict:
        """Analyze seasonality patterns"""
        try:
            # Use provided data or current data
            df = data if data is not None else self._current_data
            if df is None:
                raise ValueError("No data available for analysis")
                
            return self._forecaster.analyze_seasonality(df)
        except Exception as e:
            logger.error(f"Error analyzing seasonality: {str(e)}")
            return {}

# Create a singleton instance
_instance = None

def get_data_manager() -> DataManager:
    """Get the data manager instance"""
    global _instance
    if _instance is None:
        _instance = DataManager()
    return _instance