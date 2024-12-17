from prophet import Prophet
import pandas as pd
from typing import Optional, Dict, Tuple
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ProphetModel:
    def __init__(self):
        self.model = None
        self.last_forecast = None
        self.metrics = {}
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for Prophet model"""
        # Prophet requires columns named 'ds' (date) and 'y' (value)
        prophet_df = df.reset_index()
        prophet_df = prophet_df.rename(columns={'Date': 'ds', 'Close': 'y'})
        return prophet_df
        
    def train(self, df: pd.DataFrame, 
              changepoint_prior_scale: float = 0.05,
              seasonality_prior_scale: float = 10,
              holidays_prior_scale: float = 10) -> None:
        """Train the Prophet model"""
        prophet_df = self.prepare_data(df)
        
        self.model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        
        # Add additional regressors if available
        if 'Volume' in df.columns:
            prophet_df['volume'] = np.log1p(df['Volume'])
            self.model.add_regressor('volume')
            
        self.model.fit(prophet_df)
        
    def forecast(self, periods: int = 30) -> Tuple[pd.DataFrame, Dict]:
        """Generate forecast and return results with components"""
        if self.model is None:
            raise ValueError("Model must be trained before forecasting")
            
        future = self.model.make_future_dataframe(periods=periods)
        
        # Add regressors to future dataframe if they were used in training
        if 'volume' in self.model.extra_regressors:
            # For simplicity, use the mean volume for future predictions
            future['volume'] = future['volume'].fillna(self.model.history['volume'].mean())
            
        forecast = self.model.predict(future)
        self.last_forecast = forecast
        
        components = {
            'trend': forecast['trend'],
            'weekly': forecast['weekly'] if 'weekly' in forecast.columns else None,
            'yearly': forecast['yearly'] if 'yearly' in forecast.columns else None,
            'additive_terms': forecast['additive_terms'],
        }
        
        return forecast, components
        
    def calculate_metrics(self, actual: pd.DataFrame) -> Dict:
        """Calculate forecast accuracy metrics"""
        if self.last_forecast is None:
            raise ValueError("Must generate forecast before calculating metrics")
            
        actual_df = self.prepare_data(actual)
        forecast_df = self.last_forecast[self.last_forecast['ds'].isin(actual_df['ds'])]
        
        y_true = actual_df['y'].values
        y_pred = forecast_df['yhat'].values
        
        self.metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        return self.metrics
        
    def get_changepoints(self) -> pd.DataFrame:
        """Get detected changepoints"""
        if self.model is None:
            raise ValueError("Model must be trained first")
            
        return pd.DataFrame({
            'changepoint': self.model.changepoints,
            'impact': self.model.params['delta'].mean(axis=0)
        }) 