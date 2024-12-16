"""
Data preprocessing and validation module
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from ..utils import get_logger, get_config

logger = get_logger()
config = get_config()

class DataPreprocessor:
    def __init__(self):
        """Initialize data preprocessor"""
        self._last_processed = {}
        self._data_validity = timedelta(minutes=5)
        
    def process_data(self, data: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
        """Process and validate stock data"""
        try:
            if data is None or data.empty:
                raise ValueError("No data provided for processing")
            
            # Make a copy to avoid modifying original data
            df = data.copy()
            
            # Basic cleaning
            df = self._clean_data(df)
            
            if validate:
                df = self._validate_data(df)
            
            # Add basic features
            df = self._add_basic_features(df)
            
            # Remove outliers
            df = self._remove_outliers(df)
            
            # Final validation
            if len(df) < 20:  # Minimum required for most technical indicators
                raise ValueError("Insufficient data points after processing")
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise
            
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the data"""
        try:
            df = data.copy()
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            existing_columns = [col.lower() for col in df.columns]
            missing_columns = [col for col in required_columns if col not in existing_columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Rename columns to lowercase
            df.columns = df.columns.str.lower()
            
            # Remove rows with NaN values
            df = df.dropna(subset=required_columns)
            
            # Remove duplicate indices
            df = df[~df.index.duplicated(keep='first')]
            
            # Sort by date
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise
            
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate data consistency"""
        try:
            df = data.copy()
            
            # Validate price consistency
            df = df[
                (df['open'] > 0) & 
                (df['high'] > 0) & 
                (df['low'] > 0) & 
                (df['close'] > 0) &
                (df['volume'] >= 0)
            ]
            
            # Validate high/low consistency
            df = df[
                (df['high'] >= df['open']) & 
                (df['high'] >= df['close']) & 
                (df['low'] <= df['open']) & 
                (df['low'] <= df['close'])
            ]
            
            return df
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            raise
            
    def _add_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic features to the data"""
        try:
            df = data.copy()
            
            # Price changes
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log1p(df['returns'])
            
            # Volatility
            df['volatility'] = df['returns'].rolling(window=20, min_periods=1).std()
            
            # Volume features
            if 'volume' in df.columns:
                df['volume_ma'] = df['volume'].rolling(window=20, min_periods=1).mean()
                df['relative_volume'] = df['volume'] / df['volume_ma']
            
            # Trading ranges
            df['daily_range'] = df['high'] - df['low']
            df['daily_range_pct'] = df['daily_range'] / df['close']
            
            # Gap analysis
            df['gap'] = df['open'] - df['close'].shift(1)
            df['gap_pct'] = df['gap'] / df['close'].shift(1)
            
            # Fill NaN values
            for col in df.columns:
                if col not in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].fillna(method='bfill').fillna(method='ffill').fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding basic features: {str(e)}")
            raise
            
    def _remove_outliers(self, data: pd.DataFrame, n_sigmas: float = 3) -> pd.DataFrame:
        """Remove outliers using the IQR method"""
        try:
            df = data.copy()
            
            # Price columns to check for outliers
            price_columns = ['open', 'high', 'low', 'close']
            
            for col in price_columns:
                # Calculate IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define bounds
                lower_bound = Q1 - n_sigmas * IQR
                upper_bound = Q3 + n_sigmas * IQR
                
                # Replace outliers with boundary values
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
            
            return df
            
        except Exception as e:
            logger.error(f"Error removing outliers: {str(e)}")
            raise
            
    def analyze_data_quality(self, data: pd.DataFrame) -> Dict:
        """Analyze data quality and generate report"""
        try:
            report = {
                'total_rows': len(data),
                'date_range': {
                    'start': data.index.min(),
                    'end': data.index.max(),
                    'trading_days': len(data)
                },
                'missing_values': data.isnull().sum().to_dict(),
                'price_stats': {
                    'min_price': data['close'].min(),
                    'max_price': data['close'].max(),
                    'avg_price': data['close'].mean(),
                    'volatility': data['close'].pct_change().std()
                },
                'volume_stats': {
                    'min_volume': data['volume'].min(),
                    'max_volume': data['volume'].max(),
                    'avg_volume': data['volume'].mean()
                },
                'data_quality': {
                    'price_gaps': (data['close'].shift(1) - data['open']).abs().mean(),
                    'price_consistency': (
                        (data['high'] >= data['low']).all() and
                        (data['high'] >= data['open']).all() and
                        (data['high'] >= data['close']).all()
                    )
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error analyzing data quality: {str(e)}")
            return {} 