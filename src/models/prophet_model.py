from typing import List, Dict
import pandas as pd
from prophet import Prophet
from prophet.holidays import get_holiday_names, add_country_holidays

class ProphetModel:
    def __init__(self):
        self.model = None
        self.forecast = None
        self.metrics = None
        self.selected_features = None
    
    def _add_custom_events(self, model):
        """Add custom events and holidays that may affect stock prices."""
        # Add US holidays
        model.add_country_holidays(country_name='US')
        
        # Add custom events (e.g. earnings dates)
        custom_events = pd.DataFrame({
            'holiday': 'earnings',
            'ds': pd.to_datetime([
                '2024-01-25', '2024-04-25',  # Approximate earnings dates
                '2024-07-25', '2024-10-25'
            ]),
            'lower_window': -2,
            'upper_window': 2
        })
        model.add_holidays(custom_events)
    
    def fit(self, df: pd.DataFrame, features: List[str] = None):
        """Fit the Prophet model with optimized configuration.
        
        Args:
            df: Input dataframe with ds and y columns
            features: List of additional regressor features
        """
        # Add floor and cap for logistic growth
        df['floor'] = df['y'].min() * 0.9
        df['cap'] = df['y'].max() * 1.1
        
        # Initialize Prophet with optimized parameters
        self.model = Prophet(
            growth='logistic',  # Use logistic growth with bounds
            interval_width=0.95,  # Wider confidence intervals
            changepoint_prior_scale=0.15,  # More flexible trend changes
            seasonality_prior_scale=15,  # Stronger seasonality
            seasonality_mode='multiplicative',  # Better for financial data
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            holidays_prior_scale=0.25  # Control holiday effects
        )
        
        # Add custom weekly seasonality with Fourier terms
        self.model.add_seasonality(
            name='weekly',
            period=7,
            fourier_order=5,
            prior_scale=10
        )
        
        # Add monthly seasonality with higher order
        self.model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=10,  # Increased order for better pattern capture
            prior_scale=10
        )
        
        # Add quarterly seasonality for earnings effects
        self.model.add_seasonality(
            name='quarterly',
            period=91.25,
            fourier_order=5,
            prior_scale=15
        )
        
        # Add holidays and custom events
        self._add_custom_events(self.model)
        
        # Add validated regressors with regularization
        if features:
            for feature in features:
                self.model.add_regressor(
                    feature,
                    standardize=True,
                    mode='additive',
                    prior_scale=0.1  # Add regularization
                )
        
        # Fit the model
        self.model.fit(df)