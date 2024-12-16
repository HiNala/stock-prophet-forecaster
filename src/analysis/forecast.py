"""
Time series forecasting module using Prophet with enhanced features
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics  # Add Prophet diagnostics imports
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import joblib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools
from ..utils import get_logger, get_config
import warnings
import sys
import contextlib
import time
import logging

# Configure logging to handle cmdstanpy
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

logger = get_logger()
config = get_config()

class suppress_stdout_stderr(object):
    """Context manager for suppressing stdout and stderr."""
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

def cache_result(func):
    """Decorator for caching function results"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        cache_key = f"{func.__name__}_{hash(str(args))}_{hash(str(kwargs))}"
        return self._get_cached_data(cache_key, lambda: func(self, *args, **kwargs))
    return wrapper

class Forecaster:
    def __init__(self):
        """Initialize forecaster with optimized settings"""
        self._cache = {}
        self._max_cache_size = 100
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        self._model = None
        self._last_fit = None
        self._last_train_data = None
        self._scaler = MinMaxScaler()
        self._model_validity = timedelta(hours=1)
        self._models_dir = os.path.join('output', 'models')
        self._forecasts = {}  # Store forecasts for multiple tickers
        
        # Create models directory if it doesn't exist
        os.makedirs(self._models_dir, exist_ok=True)

    def generate_forecasts_for_tickers(self, tickers: str, data_dict: Dict[str, pd.DataFrame], 
                                     days: int = 30, confidence: float = 0.95) -> Dict[str, Tuple[pd.DataFrame, Dict]]:
        """Generate forecasts for multiple tickers in parallel
        
        Args:
            tickers: Comma-separated string of stock tickers
            data_dict: Dictionary mapping ticker symbols to their respective DataFrames
            days: Number of days to forecast
            confidence: Confidence level for prediction intervals
            
        Returns:
            Dictionary mapping ticker symbols to their (forecast, metrics) tuples
        """
        try:
            # Parse tickers
            ticker_list = [t.strip().upper() for t in tickers.split(',')]
            logger.info(f"Generating forecasts for tickers: {ticker_list}")
            
            # Create a dictionary to store results
            results = {}
            
            # Process tickers in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(len(ticker_list), 8)) as executor:
                future_to_ticker = {
                    executor.submit(self._process_single_ticker, ticker, data_dict.get(ticker), days, confidence): ticker
                    for ticker in ticker_list if ticker in data_dict
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        forecast, metrics = future.result()
                        results[ticker] = (forecast, metrics)
                        logger.info(f"Completed forecast for {ticker}")
                    except Exception as e:
                        logger.error(f"Failed to process ticker {ticker}: {str(e)}")
                        results[ticker] = (pd.DataFrame(), {})
            
            # Store forecasts for later use
            self._forecasts = {ticker: result[0] for ticker, result in results.items()}
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating multiple forecasts: {str(e)}")
            return {}

    def _process_single_ticker(self, ticker: str, data: pd.DataFrame, days: int, confidence: float) -> Tuple[pd.DataFrame, Dict]:
        """Process a single ticker and generate its forecast
        
        Args:
            ticker: Stock ticker symbol
            data: DataFrame containing stock data
            days: Number of days to forecast
            confidence: Confidence level for prediction intervals
            
        Returns:
            Tuple of (forecast DataFrame, metrics dictionary)
        """
        try:
            if data is None or data.empty:
                logger.error(f"No data provided for ticker {ticker}")
                return pd.DataFrame(), {}
            
            # Create a separate model instance for this ticker
            model_path = os.path.join(self._models_dir, f'prophet_model_{ticker}.pkl')
            
            # Generate forecast using the base method
            forecast, metrics = self.generate_forecast(data, days, confidence)
            
            # Add ticker information to the forecast
            if not forecast.empty:
                forecast['ticker'] = ticker
            
            return forecast, metrics
            
        except Exception as e:
            logger.error(f"Error processing ticker {ticker}: {str(e)}")
            return pd.DataFrame(), {}

    def get_forecast(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get the stored forecast for a specific ticker"""
        return self._forecasts.get(ticker)

    def get_all_forecasts(self) -> Dict[str, pd.DataFrame]:
        """Get all stored forecasts"""
        return self._forecasts.copy()

    def clear_forecasts(self):
        """Clear stored forecasts"""
        self._forecasts.clear()

    def cleanup(self):
        """Clean up resources"""
        self._thread_pool.shutdown(wait=False)
        self._cache.clear()
        self._model = None
        
    def _clear_cache(self):
        """Clear cache if it gets too large"""
        if len(self._cache) > self._max_cache_size:
            self._cache.clear()
            
    def _get_cached_data(self, key, data_fn):
        """Get data from cache or compute it"""
        if key not in self._cache:
            self._cache[key] = data_fn()
            self._clear_cache()
        return self._cache[key]
        
    @cache_result
    def generate_forecast(self, data: pd.DataFrame, days: int = 30,
                         confidence: float = 0.95) -> Tuple[pd.DataFrame, Dict]:
        """Generate price forecast with improved model configuration"""
        try:
            # Validate inputs
            if data is None or data.empty:
                logger.error("No data provided for forecast generation")
                return pd.DataFrame(), {}
                
            days = max(1, min(days, 365))
            confidence = max(0.5, min(confidence, 0.99))
            
            # Check if we need to retrain the model
            model_path = os.path.join(self._models_dir, 'prophet_model.pkl')
            if (self._model is None or
                self._last_fit is None or
                datetime.now() - self._last_fit > self._model_validity):
                
                logger.debug("Fitting new model...")
                # Fit model using consolidated training method
                if not self._fit_model(data, confidence):
                    logger.error("Failed to fit model")
                    return pd.DataFrame(), {}
                
                # Save model
                try:
                    joblib.dump(self._model, model_path)
                except Exception as e:
                    logger.warning(f"Failed to save model: {str(e)}")
            else:
                try:
                    # Load existing model
                    logger.debug("Loading existing model...")
                    self._model = joblib.load(model_path)
                except Exception as e:
                    logger.warning(f"Failed to load model, fitting new one: {str(e)}")
                    if not self._fit_model(data, confidence):
                        logger.error("Failed to fit model")
                        return pd.DataFrame(), {}
            
            # Generate forecast with parallel feature calculation
            logger.debug("Generating forecast...")
            forecast = self._generate_forecast_parallel(days)
            if forecast is None:
                logger.error("Forecast generation returned None")
                return pd.DataFrame(), {}
            if forecast.empty:
                logger.error("Forecast generation returned empty DataFrame")
                return pd.DataFrame(), {}
            
            logger.debug(f"Generated forecast columns: {forecast.columns.tolist()}")
            
            # Ensure forecast has required columns
            required_columns = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
            missing_columns = [col for col in required_columns if col not in forecast.columns]
            if missing_columns:
                logger.error(f"Missing required columns in forecast: {missing_columns}")
                return pd.DataFrame(), {}
            
            # Calculate metrics
            metrics = self._calculate_forecast_metrics(data, forecast)
            
            return forecast, metrics
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            return pd.DataFrame(), {}
            
    def _prepare_prophet_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for Prophet with enhanced features and validation"""
        try:
            logger.debug(f"Initial data shape: {data.shape}")
            logger.debug(f"Initial columns: {data.columns.tolist()}")
            
            # Create a clean copy and ensure datetime index
            data = data.copy()
            data = data.reset_index()
            date_col = 'Date' if 'Date' in data.columns else data.columns[0]
            logger.debug(f"Using date column: {date_col}")
            
            # Validate date column and convert to datetime
            data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
            if data[date_col].isna().any():
                logger.error("Found NaN values in date column after conversion")
                data = data.dropna(subset=[date_col])
                
            data = data.sort_values(date_col)
            
            # Create Prophet dataframe with explicit datetime validation
            df = pd.DataFrame({'ds': data[date_col]})
            df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
            if df['ds'].isna().any():
                logger.error("Found NaN values in 'ds' column after creation")
                df = df.dropna(subset=['ds'])
            
            # Add target variable with validation
            target_col = 'close_original' if 'close_original' in data.columns else 'close'
            if data[target_col].isna().any():
                logger.error(f"Found NaN values in target column: {target_col}")
                data[target_col] = data[target_col].ffill().bfill()
            
            df['y'] = np.log(data[target_col])
            
            # Initialize last_values dictionary
            self._last_values = {}
            
            # Add technical indicators with validation
            df = self._add_technical_indicators(data, df)
            
            # Calculate trend features with validation
            df = self._calculate_trend_features(df)
            
            # Add calendar features
            df = self._add_calendar_features(df, df['ds'])
            
            # Final validation
            logger.debug("Performing final validation...")
            logger.debug(f"DataFrame shape: {df.shape}")
            logger.debug(f"Columns: {df.columns.tolist()}")
            
            # Ensure ds column is still datetime and not modified
            df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
            if df['ds'].isna().any():
                logger.error("Found NaN values in 'ds' column during final validation")
                df = df.dropna(subset=['ds'])
            
            # Check for any remaining NaN values
            nan_cols = df.columns[df.isna().any()].tolist()
            if nan_cols:
                logger.error(f"Found NaN values in columns: {nan_cols}")
                for col in nan_cols:
                    logger.debug(f"NaN count in {col}: {df[col].isna().sum()}")
                    df[col] = df[col].ffill().bfill().fillna(0)
            
            # Final Prophet requirements check
            required_cols = ['ds', 'y']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required Prophet columns: {missing_cols}")
            
            # Verify Prophet requirements
            logger.debug("Verifying Prophet requirements...")
            logger.debug(f"'ds' column type: {df['ds'].dtype}")
            logger.debug(f"'y' column type: {df['y'].dtype}")
            logger.debug(f"First few rows of Prophet data:\n{df[['ds', 'y']].head()}")
            
            if not pd.api.types.is_datetime64_any_dtype(df['ds']):
                raise ValueError("'ds' column is not datetime type")
            if not pd.api.types.is_numeric_dtype(df['y']):
                raise ValueError("'y' column is not numeric type")
            
            logger.debug("Data preparation completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error preparing Prophet data: {str(e)}")
            raise
            
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI efficiently"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)  # Fill NaN with neutral RSI value
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            raise
            
    def _fit_model(self, data: pd.DataFrame, confidence: float = 0.95) -> bool:
        """Fit Prophet model with optimized parameters and validation"""
        try:
            start_time = time.time()
            logger.debug("Starting model fitting process...")
            
            # Prepare data and verify length
            initial_length = len(data)
            df = self._prepare_prophet_data(data)
            if df is None or df.empty:
                logger.error("Failed to prepare data for Prophet")
                return False
            
            # Verify no rows were lost during preparation
            if len(df) != initial_length:
                logger.warning(f"Dataset length changed during preparation: {initial_length} -> {len(df)}")
                
            prep_time = time.time()
            logger.debug(f"Data preparation completed in {prep_time - start_time:.2f} seconds")
            
            # Calculate dynamic model parameters based on data characteristics
            volatility = df['y'].pct_change().rolling(30).std().mean()
            changepoint_prior = max(0.01, min(0.1, volatility * 5))  # Dynamic scaling
            
            # Detect and handle outliers in target variable
            y_zscore = zscore(df['y'])
            outliers = (y_zscore.abs() > 3).sum()
            if outliers > 0:
                logger.warning(f"Detected {outliers} outliers in target variable")
                window = 5
                df['y'] = df['y'].where(y_zscore.abs() <= 3, 
                                      df['y'].rolling(window=window, center=True, min_periods=1).median())
            
            # Store original prices for later use
            self._original_prices = np.exp(df['y'].copy())
            
            # Calculate trend metrics with improved estimation
            recent_returns = pd.Series(np.diff(np.log(self._original_prices[-30:])))
            recent_volatility = recent_returns.ewm(span=10).std().iloc[-1] * np.sqrt(252)
            recent_trend = recent_returns.ewm(span=10).mean().iloc[-1] * 252
            trend_consistency = df['trend_consistency'].tail(30).mean() if 'trend_consistency' in df.columns else 0.5
            trend_strength = abs(recent_trend)
            
            # Define potential regressors with their characteristics
            potential_regressors = {
                'trend_consistency': {'prior_scale': 0.8, 'mode': 'multiplicative'},
                'price_momentum': {'prior_scale': 0.7, 'mode': 'multiplicative'},
                'volatility': {'prior_scale': 0.5, 'mode': 'multiplicative'},
                'regime_strength': {'prior_scale': 0.6, 'mode': 'multiplicative'},
                'volume_trend': {'prior_scale': 0.4, 'mode': 'additive'},
                'rsi': {'prior_scale': 0.3, 'mode': 'additive'}
            }
            
            # Prepare feature selection data
            feature_data = pd.DataFrame()
            for feature in potential_regressors.keys():
                if feature in df.columns:
                    feature_data[feature] = df[feature]
            
            if not feature_data.empty:
                # Calculate feature correlations with target
                correlations = abs(feature_data.corrwith(df['y']))
                
                # Calculate feature stability (lower variance is more stable)
                stability = 1 / feature_data.std()
                
                # Calculate feature importance score
                importance_scores = correlations * stability
                
                # Select top features (maximum 4)
                top_features = importance_scores.nlargest(4).index.tolist()
                
                # Update regressors dictionary with selected features
                important_regressors = {
                    feature: potential_regressors[feature]
                    for feature in top_features
                }
            else:
                important_regressors = {}
            
            logger.debug(f"Selected important regressors: {list(important_regressors.keys())}")
            
            # Create final Prophet DataFrame
            prophet_columns = ['ds', 'y'] + list(important_regressors.keys())
            if trend_strength > 0.6:
                growth = 'logistic'
                cap_multiplier = 1.2 + (trend_strength * 0.2)  # More conservative multiplier
                df['cap'] = df['y'].max() * cap_multiplier
                df['floor'] = df['y'].min() * 0.8  # Higher floor for stability
                prophet_columns.extend(['cap', 'floor'])
            else:
                growth = 'linear'
            
            prophet_df = df[prophet_columns].copy()
            
            # Configure model with optimized parameters
            self._model = Prophet(
                growth=growth,
                changepoint_prior_scale=changepoint_prior,
                changepoint_range=0.9,
                n_changepoints=15,
                interval_width=confidence,
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False,
                seasonality_mode='multiplicative' if trend_strength > 0.6 else 'additive',
                seasonality_prior_scale=3  # Reduced from 5
            )
            
            # Add custom seasonalities with reduced complexity
            if len(df) >= 30:
                self._model.add_seasonality(
                    name='monthly',
                    period=30.5,
                    fourier_order=3,
                    prior_scale=0.1
                )
            
            # Add validated regressors
            for feature, params in important_regressors.items():
                self._model.add_regressor(
                    feature,
                    prior_scale=params['prior_scale'],
                    mode=params['mode']
                )
            
            # Store the data for later use
            self._last_train_data = prophet_df.copy()
            
            # Fit model with validation
            with suppress_stdout_stderr():
                self._model.fit(prophet_df)
            
            # Verify model fitting was successful
            if not hasattr(self._model, 'history'):
                logger.error("Model fitting failed - no history attribute")
                return False
            
            # Validate model performance with expanded cross-validation
            try:
                cv_results = cross_validation(
                    self._model,
                    initial='180 days',
                    period='30 days',
                    horizon='30 days',
                    parallel="threads"
                )
                metrics = performance_metrics(cv_results)
                logger.debug(f"Cross-validation metrics:\n{metrics}")
                
                # Store cross-validation results for later use
                self._cv_results = cv_results
                self._cv_metrics = metrics
                
            except Exception as e:
                logger.warning(f"Error during cross-validation: {str(e)}")
            
            fit_time = time.time()
            logger.debug(f"Model fitting completed in {fit_time - prep_time:.2f} seconds")
            
            self._last_fit = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            return False
            
    def _get_feature_prior_scale(self, df: pd.DataFrame, feature: str, group: str) -> float:
        """Calculate appropriate prior scale for a feature"""
        try:
            if group == 'regime':
                return 0.5  # Higher impact for regime features
            elif group == 'volatility':
                return 0.3 * (1 + df[feature].std())  # Adaptive prior scale
            elif group == 'momentum':
                return 0.4  # Moderate impact for momentum
            elif group == 'technical':
                return 0.1  # Lower impact for technical indicators
            elif group == 'volume':
                correlation = abs(df[feature].corr(df['y']))
                return 0.2 * (1 + correlation)  # Higher impact for correlated features
            else:
                return 0.3  # Default value
        except Exception as e:
            logger.warning(f"Error calculating prior scale for {feature}: {str(e)}")
            return 0.3  # Default fallback value
            
    def _generate_forecast_parallel(self, days: int) -> pd.DataFrame:
        """Generate forecast with parallel feature calculation and enhanced confidence intervals"""
        try:
            # Store the list of regressors used in training
            if not hasattr(self, '_last_train_data') or self._last_train_data is None:
                logger.error("No training data available for forecast generation")
                return None

            # Get the list of regressors from the model
            regressors = [reg for reg in self._model.extra_regressors.keys()]
            logger.debug(f"Required regressors for forecast: {regressors}")
            
            # Create future dataframe with more granular dates
            future = self._model.make_future_dataframe(
                periods=days,
                freq='D',
                include_history=True
            )
            
            # Add caps and floors if using logistic growth
            if self._model.growth == 'logistic':
                future['cap'] = self._last_train_data['cap'].max()
                future['floor'] = self._last_train_data['floor'].min()
            
            # Calculate recent trend metrics with improved estimation
            recent_window = min(30, len(self._original_prices))
            recent_returns = pd.Series(np.diff(np.log(self._original_prices[-recent_window:])))
            
            # Use EWMA for trend calculation
            recent_trend = recent_returns.ewm(span=10).mean().iloc[-1] * 252
            trend_volatility = recent_returns.ewm(span=20).std().iloc[-1] * np.sqrt(252)
            
            # Get trend consistency and regime information
            trend_consistency = self._last_values.get('trend_consistency', 0.5)
            regime_strength = self._last_values.get('regime_strength', 0.5)
            
            # Project features with improved stability
            for feature in regressors:
                try:
                    if feature not in self._last_train_data.columns:
                        logger.error(f"Required regressor {feature} not found in training data")
                        return None
                        
                    last_value = self._last_train_data[feature].iloc[-1]
                    feature_std = self._last_train_data[feature].std()
                    
                    # Calculate days from start for each future date
                    days_out = (future['ds'] - future['ds'].min()).dt.days.values
                    max_days = days_out.max()
                    
                    if feature in ['trend_consistency', 'price_momentum', 'volatility']:
                        # Calculate adaptive decay rate based on feature characteristics
                        base_rate = 0.002  # Slower base rate for more stability
                        vol_factor = np.clip(1 + trend_volatility, 0.8, 1.2)
                        consistency_factor = np.clip(trend_consistency, 0.6, 1.0)
                        
                        # Calculate mean reversion target
                        historical_mean = self._last_train_data[feature].mean()
                        reversion_strength = 1 - consistency_factor
                        
                        # Generate feature projection with mean reversion
                        if recent_trend > 0 and trend_consistency > 0.7:
                            # Uptrend scenario: slower decay, higher persistence
                            decay_rate = base_rate / (vol_factor * consistency_factor)
                            growth_factor = 1 + (days_out / max_days * 0.001 * recent_trend * trend_consistency)
                            values = last_value * np.exp(-decay_rate * days_out) * growth_factor
                        else:
                            # Mean reversion scenario
                            decay_rate = base_rate * vol_factor / consistency_factor
                            reversion_path = last_value + (historical_mean - last_value) * (1 - np.exp(-decay_rate * days_out))
                            values = reversion_path
                        
                        # Apply smoothing to reduce noise
                        future[feature] = pd.Series(values).rolling(
                            window=5, min_periods=1, center=True
                        ).mean()
                        
                        # Add bounds to prevent extreme values
                        lower_bound = max(historical_mean - 2 * feature_std, 0)
                        upper_bound = historical_mean + 2 * feature_std
                        future[feature] = future[feature].clip(lower_bound, upper_bound)
                        
                    else:
                        # For other features, use weighted mean reversion
                        historical_mean = self._last_train_data[feature].mean()
                        reversion_rate = 0.03  # Slower mean reversion
                        
                        # Calculate adaptive weights based on regime strength
                        regime_weight = np.clip(regime_strength, 0.3, 0.7)
                        mean_weight = 1 - regime_weight
                        
                        # Generate feature projection
                        reversion_target = (historical_mean * mean_weight + last_value * regime_weight)
                        future[feature] = last_value + (reversion_target - last_value) * (1 - np.exp(-reversion_rate * days_out))
                        
                        # Smooth the projection
                        future[feature] = future[feature].rolling(
                            window=3, min_periods=1, center=True
                        ).mean()
                    
                except Exception as e:
                    logger.error(f"Error projecting feature {feature}: {str(e)}")
                    return None
            
            # Generate forecast with increased samples and cross-validation
            try:
                with suppress_stdout_stderr():
                    forecast = self._model.predict(future)
                
                # Transform predictions back from log space
                forecast['yhat'] = np.exp(forecast['yhat'])
                forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
                forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])
                
                # Apply forecast smoothing
                forecast['yhat'] = forecast['yhat'].rolling(
                    window=3, center=True, min_periods=1
                ).mean()
                
                # Calculate dynamic prediction intervals
                if hasattr(self, '_cv_metrics'):
                    try:
                        # Use cross-validation metrics to adjust intervals
                        horizon_metrics = self._cv_metrics.groupby('horizon')[['mape', 'coverage']].mean()
                        
                        # Calculate dynamic volatility
                        returns = pd.Series(np.diff(np.log(self._original_prices[-60:])))
                        vol = returns.ewm(span=20, adjust=False).std().iloc[-1] * np.sqrt(252)
                        
                        # Calculate base width using volatility and CV metrics
                        base_width = vol * 0.7  # Reduced base width
                        
                        # Adjust intervals based on forecast horizon
                        days_out = np.arange(len(forecast)) / len(forecast)
                        
                        # Calculate horizon-based width multiplier
                        width_multiplier = np.ones_like(days_out)
                        for i, day in enumerate(days_out):
                            horizon_day = min(30, int(day * 30))  # Map to CV horizon
                            if horizon_day in horizon_metrics.index:
                                # Use CV metrics to adjust width
                                coverage_error = abs(0.95 - horizon_metrics.loc[horizon_day, 'coverage'])
                                mape = horizon_metrics.loc[horizon_day, 'mape']
                                width_multiplier[i] = 1 + (coverage_error * 0.5) + (mape * 0.01)
                        
                        # Apply regime-based adjustments
                        if trend_consistency > 0.7 and recent_trend > 0:
                            width_multiplier *= 0.9  # Tighter intervals in strong uptrends
                        elif trend_consistency < 0.3:
                            width_multiplier *= 1.1  # Wider intervals in uncertain regimes
                        
                        # Calculate final intervals
                        width = base_width * width_multiplier * (1 + days_out * 0.005)  # Slower widening
                        forecast['yhat_lower'] = forecast['yhat'] * np.exp(-width)
                        forecast['yhat_upper'] = forecast['yhat'] * np.exp(width)
                        
                        # Ensure minimum spread
                        min_spread = 0.01  # 1% minimum spread
                        spread = (forecast['yhat_upper'] - forecast['yhat_lower']) / forecast['yhat']
                        too_tight = spread < min_spread
                        if too_tight.any():
                            adj_width = min_spread / 2
                            forecast.loc[too_tight, 'yhat_lower'] = forecast.loc[too_tight, 'yhat'] * (1 - adj_width)
                            forecast.loc[too_tight, 'yhat_upper'] = forecast.loc[too_tight, 'yhat'] * (1 + adj_width)
                        
                    except Exception as e:
                        logger.warning(f"Error calculating dynamic intervals: {str(e)}")
                
                return forecast
                
            except Exception as e:
                logger.error(f"Error in Prophet prediction: {str(e)}")
                return None
            
        except Exception as e:
            logger.error(f"Error generating parallel forecast: {str(e)}")
            return None
            
    def _calculate_forecast_metrics(self, data: pd.DataFrame,
                                  forecast: pd.DataFrame) -> Dict:
        """Calculate forecast metrics efficiently"""
        try:
            if data is None or data.empty or forecast is None or forecast.empty:
                return {
                    'pred_price': "N/A",
                    'upper_bound': "N/A",
                    'lower_bound': "N/A",
                    'confidence': "N/A",
                    'trend': "N/A",
                    'change': "N/A",
                    'accuracy': "N/A",
                    'volatility': "N/A"
                }
            
            # Calculate trend metrics
            trend = forecast['trend'].diff().mean()
            trend_direction = 'Upward' if trend > 0 else 'Downward' if trend < 0 else 'Sideways'
            trend_strength = abs(trend)
            
            # Calculate volatility metrics
            volatility = (forecast['yhat_upper'] - forecast['yhat_lower']).mean()
            relative_volatility = volatility / forecast['yhat'].mean() * 100
            
            # Calculate forecast range
            last_price = data['close'].iloc[-1]
            forecast_end = forecast['yhat'].iloc[-1]
            forecast_change = (forecast_end - last_price) / last_price * 100
            
            # Calculate confidence metrics
            confidence_width = (forecast['yhat_upper'] - forecast['yhat_lower']).mean()
            relative_confidence = confidence_width / forecast['yhat'].mean() * 100
            
            # Calculate accuracy metrics (using historical data)
            try:
                # Convert data index to timezone-naive datetime
                data_index = pd.to_datetime(data.index).tz_localize(None)
                forecast_dates = pd.to_datetime(forecast['ds']).dt.tz_localize(None)
                
                # Find overlapping dates
                common_dates = data_index.intersection(forecast_dates)
                
                if len(common_dates) > 0:
                    # Get corresponding actual and predicted values
                    actual_values = data.loc[common_dates, 'close']
                    predicted_values = forecast.set_index('ds').loc[common_dates, 'yhat']
                    
                    # Calculate MAPE
                    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
                else:
                    mape = np.nan
                    
            except Exception as e:
                logger.warning(f"Error calculating accuracy metrics: {str(e)}")
                mape = np.nan
            
            # Format metrics for display
            return {
                'pred_price': f"${forecast_end:.2f}",
                'upper_bound': f"${forecast['yhat_upper'].iloc[-1]:.2f}",
                'lower_bound': f"${forecast['yhat_lower'].iloc[-1]:.2f}",
                'confidence': f"{relative_confidence:.1f}%",
                'trend': f"{trend_direction} ({trend_strength:.2f})",
                'change': f"{forecast_change:+.1f}%",
                'accuracy': f"{100-mape:.1f}%" if not np.isnan(mape) else "N/A",
                'volatility': f"{relative_volatility:.1f}%"
            }
            
        except Exception as e:
            logger.error(f"Error calculating forecast metrics: {str(e)}")
            return {
                'pred_price': "N/A",
                'upper_bound': "N/A",
                'lower_bound': "N/A",
                'confidence': "N/A",
                'trend': "N/A",
                'change': "N/A",
                'accuracy': "N/A",
                'volatility': "N/A"
            }
            
    def get_components(self) -> Optional[Dict]:
        """Get forecast components for analysis"""
        try:
            if self._model is None or self._last_fit is None:
                return None
                
            # Check if model is still fresh
            if datetime.now() - self._last_fit > self._model_validity:
                return None
                
            # Get components from Prophet model
            components = {
                'trend': self._model.trend,
                'seasonality': {
                    'daily': self._model.daily_seasonality,
                    'weekly': self._model.weekly_seasonality,
                    'yearly': self._model.yearly_seasonality,
                    'monthly': True  # Custom added seasonality
                },
                'changepoints': self._model.changepoints,
                'changepoint_prior': self._model.changepoint_prior_scale,
                'seasonality_prior': self._model.seasonality_prior_scale,
                'holidays_prior': self._model.holidays_prior_scale
            }
            
            return components
            
        except Exception as e:
            logger.error(f"Error getting components: {str(e)}")
            return None
            
    def analyze_seasonality(self, data: pd.DataFrame) -> Dict:
        """Analyze seasonality patterns efficiently"""
        try:
            # Create cache key
            cache_key = f"seasonality_{data.index[0]}_{data.index[-1]}"
            
            def calculate_seasonality():
                # Convert to datetime index if needed
                df = data.copy()
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                
                # Calculate daily patterns
                df['hour'] = df.index.hour
                daily_pattern = df.groupby('hour')['close'].mean()
                
                # Calculate weekly patterns
                df['dayofweek'] = df.index.dayofweek
                weekly_pattern = df.groupby('dayofweek')['close'].mean()
                
                # Calculate monthly patterns
                df['month'] = df.index.month
                monthly_pattern = df.groupby('month')['close'].mean()
                
                # Calculate strength of seasonality
                daily_strength = daily_pattern.std() / daily_pattern.mean()
                weekly_strength = weekly_pattern.std() / weekly_pattern.mean()
                monthly_strength = monthly_pattern.std() / monthly_pattern.mean()
                
                return {
                    'daily_pattern': daily_pattern,
                    'weekly_pattern': weekly_pattern,
                    'monthly_pattern': monthly_pattern,
                    'seasonality_strength': {
                        'daily': daily_strength,
                        'weekly': weekly_strength,
                        'monthly': monthly_strength
                    }
                }
                
            return self._get_cached_data(cache_key, calculate_seasonality)
            
        except Exception as e:
            logger.error(f"Error analyzing seasonality: {str(e)}")
            raise
            
    def validate_forecast(self, data: pd.DataFrame, forecast: pd.DataFrame) -> Dict:
        """Enhanced forecast validation with multiple metrics"""
        try:
            # Use last 20% of data for validation
            split_idx = int(len(data) * 0.8)
            train_data = data.iloc[:split_idx]
            test_data = data.iloc[split_idx:]
            
            # Generate forecast for validation period
            val_forecast, _ = self.generate_forecast(
                train_data,
                days=len(test_data),
                confidence=0.95
            )
            
            if val_forecast is None or val_forecast.empty:
                return {
                    'accuracy': 0.0,
                    'metrics': {
                        'mape': float('inf'),
                        'rmse': float('inf'),
                        'direction_accuracy': 0.0,
                        'within_bounds': 0.0
                    }
                }
            
            # Calculate multiple error metrics
            actual_values = test_data['close'].values
            predicted_values = val_forecast['yhat'].iloc[-len(test_data):].values
            lower_bounds = val_forecast['yhat_lower'].iloc[-len(test_data):].values
            upper_bounds = val_forecast['yhat_upper'].iloc[-len(test_data):].values
            
            # MAPE calculation
            mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
            
            # RMSE calculation
            rmse = np.sqrt(np.mean((actual_values - predicted_values)**2))
            
            # Direction accuracy
            actual_direction = np.sign(np.diff(actual_values))
            predicted_direction = np.sign(np.diff(predicted_values))
            direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
            
            # Prediction interval coverage
            within_bounds = np.mean(
                (actual_values >= lower_bounds) &
                (actual_values <= upper_bounds)
            ) * 100
            
            # Calculate overall accuracy score (weighted combination of metrics)
            accuracy_score = (
                0.4 * (100 - min(mape, 100)) +  # MAPE contribution (inversed)
                0.3 * direction_accuracy +       # Direction accuracy contribution
                0.2 * within_bounds +           # Prediction interval coverage
                0.1 * (100 - min(rmse / actual_values.mean() * 100, 100))  # Normalized RMSE contribution
            )
            
            return {
                'accuracy': accuracy_score,
                'metrics': {
                    'mape': mape,
                    'rmse': rmse,
                    'direction_accuracy': direction_accuracy,
                    'within_bounds': within_bounds
                }
            }
            
        except Exception as e:
            logger.error(f"Error validating forecast: {str(e)}")
            return {
                'accuracy': 0.0,
                'metrics': {
                    'mape': float('inf'),
                    'rmse': float('inf'),
                    'direction_accuracy': 0.0,
                    'within_bounds': 0.0
                }
            }

    def _calculate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-related features with robust NaN handling"""
        try:
            logger.debug("Calculating trend features...")
            
            # Store original columns for validation
            initial_columns = set(df.columns)
            logger.debug(f"Initial columns: {initial_columns}")
            
            # Define expected new feature columns
            new_feature_columns = {
                'price_momentum', 'trend_acceleration', 
                'volume_price_impact', 'trend_regime',
                'trend_consistency'  # Add trend_consistency to expected features
            }
            
            # Define all expected columns after calculation
            expected_columns = initial_columns.union(new_feature_columns)
            logger.debug(f"Expected final columns: {expected_columns}")
            
            # Track intermediate calculations and NaN values
            def track_columns(step_name: str):
                current = set(df.columns)
                new = current - initial_columns - new_feature_columns
                if new:
                    logger.warning(f"Unexpected columns after {step_name}: {new}")
                    df.drop(columns=list(new), inplace=True)
                
                # Check for NaN values
                nan_check = df.isna().sum()
                nan_cols = nan_check[nan_check > 0]
                if not nan_cols.empty:
                    logger.warning(f"NaN values found after {step_name}: {nan_cols.to_dict()}")
            
            # Calculate price momentum with explicit NaN handling
            df['price_momentum'] = (
                df['y']
                .diff(5)
                .ffill()  # Replace fillna(method='ffill')
                .bfill()  # Replace fillna(method='bfill')
                .fillna(0)
                .rolling(window=5, min_periods=1)
                .mean()
                .fillna(0)
            )
            track_columns("price_momentum")
            logger.debug(f"NaN count in price_momentum: {df['price_momentum'].isna().sum()}")
            
            # Calculate trend acceleration with careful NaN handling
            df['trend_acceleration'] = (
                df['price_momentum']
                .diff()
                .ffill()  # Replace fillna(method='ffill')
                .bfill()  # Replace fillna(method='bfill')
                .fillna(0)
                .rolling(window=5, min_periods=1, center=True)
                .mean()
                .fillna(0)
            )
            track_columns("trend_acceleration")
            logger.debug(f"NaN count in trend_acceleration: {df['trend_acceleration'].isna().sum()}")
            
            # Calculate trend consistency
            if 'returns' in df.columns:
                # Calculate the sign of returns
                returns_sign = np.sign(df['returns'].fillna(0))
                # Calculate rolling window of consistent trend direction
                window_size = 20
                df['trend_consistency'] = (
                    returns_sign.rolling(window=window_size, min_periods=1)
                    .apply(lambda x: abs(x.sum()) / len(x))
                    .fillna(0)
                )
            else:
                df['trend_consistency'] = 0.0
            track_columns("trend_consistency")
            logger.debug(f"NaN count in trend_consistency: {df['trend_consistency'].isna().sum()}")
            
            # Calculate volume price impact if possible
            if 'volume' in df.columns and 'returns' in df.columns:
                df['volume_price_impact'] = (
                    df['volume'].replace([np.inf, -np.inf], np.nan).fillna(0) * 
                    df['returns'].replace([np.inf, -np.inf], np.nan).abs().fillna(0)
                ).rolling(window=10, min_periods=1).mean().fillna(0)
            else:
                df['volume_price_impact'] = 0.0
            track_columns("volume_price_impact")
            
            # Calculate trend regime with NaN-safe operations
            if 'trend_consistency' in df.columns and 'returns' in df.columns:
                trend_mean = (
                    df['returns']
                    .fillna(0)
                    .rolling(20, min_periods=1)
                    .mean()
                    .fillna(0)
                )
                trend_consistency = df['trend_consistency'].fillna(0)
                df['trend_regime'] = np.where(
                    (trend_consistency > 0.7) & (trend_mean > 0),
                    1.0, 0.0
                )
            else:
                df['trend_regime'] = 0.0
            track_columns("trend_regime")
            
            # Final column validation
            final_columns = set(df.columns)
            missing_columns = new_feature_columns - final_columns
            if missing_columns:
                logger.error(f"Missing required feature columns: {missing_columns}")
                for col in missing_columns:
                    df[col] = 0.0
            
            extra_columns = final_columns - expected_columns
            if extra_columns:
                logger.warning(f"Removing extra columns: {extra_columns}")
                df.drop(columns=list(extra_columns), inplace=True)
            
            # Final NaN check and cleanup
            nan_check = df.isna().sum()
            nan_cols = nan_check[nan_check > 0]
            if not nan_cols.empty:
                logger.warning(f"Final NaN check - columns with NaN: {nan_cols.to_dict()}")
                df = df.fillna(0)  # Fill any remaining NaNs with 0
            
            # Verify all required columns exist and are NaN-free
            actual_columns = set(df.columns)
            if not new_feature_columns.issubset(actual_columns):
                missing = new_feature_columns - actual_columns
                raise ValueError(f"Required feature columns missing after calculation: {missing}")
            
            # Final assertion to ensure no NaNs remain
            assert df['trend_acceleration'].isna().sum() == 0, "NaN detected in trend_acceleration after cleanup!"
            assert df.isna().sum().sum() == 0, "NaN values detected in DataFrame after cleanup!"
            
            logger.debug(f"Final columns: {sorted(df.columns)}")
            logger.debug("Trend features calculation completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating trend features: {str(e)}")
            raise

    def _add_technical_indicators(self, data: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators with proper NaN handling"""
        try:
            logger.debug("Adding technical indicators...")
            
            # Handle RSI calculation first
            if 'rsi' not in data.columns:
                data['rsi'] = self._calculate_rsi(data['close'])
            
            # Add technical indicators with better scaling
            key_features = [
                'sma_20', 'sma_50', 'ema_20', 'ema_50',
                'rsi', 'volatility', 'trend_consistency',
                'regime_strength', 'volume_trend'
            ]
            
            for feature in key_features:
                if feature in data.columns:
                    if feature in ['rsi', 'trend_consistency']:
                        # RSI and trend consistency are already scaled
                        df[feature] = data[feature].ffill().bfill().fillna(50)  # Neutral value for missing data
                    else:
                        # Z-score normalization for other features
                        values = data[feature].ffill().bfill().fillna(0).values
                        if np.std(values) != 0:
                            df[feature] = (values - np.mean(values)) / np.std(values)
                        else:
                            df[feature] = 0  # If std is 0, set feature to 0
                    self._last_values[feature] = df[feature].iloc[-1]
            
            # Ensure returns and volatility exist
            if 'returns' not in data.columns:
                data['returns'] = data['close'].pct_change()
            if 'volatility' not in data.columns:
                data['volatility'] = data['returns'].rolling(window=20, min_periods=1).std()
            
            # Add momentum features with proper scaling and NaN handling
            df['returns'] = data['returns'].ffill().bfill().fillna(0)
            df['volatility'] = data['volatility'].ffill().bfill().fillna(0)
            
            # Store momentum features
            for feature in ['returns', 'volatility']:
                self._last_values[feature] = df[feature].iloc[-1]
            
            # Add volume features with log transformation and NaN handling
            if 'volume' in data.columns:
                df['volume'] = np.log1p(data['volume'])  # log1p handles zero values better
                df['volume_trend'] = df['volume'].diff(5).rolling(window=5, min_periods=1).mean()
                df['volume_momentum'] = df['volume_trend'].diff().rolling(window=5, min_periods=1).mean()
                
                for feature in ['volume', 'volume_trend', 'volume_momentum']:
                    df[feature] = df[feature].ffill().bfill().fillna(0)
                    self._last_values[feature] = df[feature].iloc[-1]
            
            # Add trend and regime indicators with proper NaN handling
            if 'sma_200' in data.columns:
                df['above_sma_200'] = (data['close'] > data['sma_200']).astype(float)
            else:
                df['above_sma_200'] = 0.5  # Neutral value if SMA200 not available
            
            df['volatility_regime'] = data.get('volatility_regime', pd.Series(index=data.index, data='medium')).map({
                'low': 0.0, 'medium': 0.5, 'high': 1.0
            }).fillna(0.5)
            
            # Final validation
            nan_cols = df.columns[df.isna().any()].tolist()
            if nan_cols:
                logger.warning(f"Found NaN values in technical indicators: {nan_cols}")
                for col in nan_cols:
                    df[col] = df[col].ffill().bfill().fillna(0)
            
            logger.debug("Technical indicators added successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            raise

    def _add_calendar_features(self, df: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
        """Add calendar-based features with validation"""
        try:
            logger.debug("Adding calendar features...")
            
            # Add basic calendar features
            for feature in ['month', 'day', 'dayofweek', 'quarter']:
                df[feature] = getattr(dates.dt, feature)
                self._last_values[feature] = df[feature].iloc[-1]
            
            # Add month start/end indicators
            df['month_start'] = dates.dt.is_month_start.astype(float)
            df['month_end'] = dates.dt.is_month_end.astype(float)
            
            # Add quarter start/end indicators
            df['quarter_start'] = (df['month'] % 3 == 1).astype(float)
            df['quarter_end'] = (df['month'] % 3 == 0).astype(float)
            
            # Store calendar features
            for feature in ['month_start', 'month_end', 'quarter_start', 'quarter_end']:
                self._last_values[feature] = df[feature].iloc[-1]
            
            logger.debug("Calendar features added successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error adding calendar features: {str(e)}")
            raise

    def _test_regressor_importance(self, df: pd.DataFrame, regressor: str, 
                                 prior_scale: float, important_regressors: Dict[str, float],
                                 base_mape: float) -> Tuple[Dict[str, float], float]:
        """Test a single regressor's importance using cross-validation"""
        try:
            # Create test DataFrame with current regressor
            test_df = df[['ds', 'y'] + list(important_regressors.keys())].copy()
            test_df[regressor] = df[regressor]
            
            # Configure test model
            test_model = Prophet(
                growth='linear',
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10,
                daily_seasonality=False,
                weekly_seasonality=True,
                interval_width=0.95
            )
            
            # Add existing important regressors
            for reg, scale in important_regressors.items():
                test_model.add_regressor(reg, prior_scale=scale)
                
            # Add new regressor
            test_model.add_regressor(regressor, prior_scale=prior_scale)
            
            # Fit model
            with suppress_stdout_stderr():
                test_model.fit(test_df)
            
            # Perform cross-validation
            cv_df = cross_validation(
                test_model,
                initial='120 days',  # Use 4 months of training data
                period='30 days',    # Test on 1-month periods
                horizon='30 days',   # Forecast 1 month ahead
                parallel="threads"   # Use parallel processing
            )
            
            # Calculate performance metrics
            metrics_df = performance_metrics(cv_df)
            current_mape = metrics_df['mape'].mean()
            
            # Update regressors if performance improves
            if current_mape < base_mape:
                logger.debug(f"Regressor {regressor} improves MAPE from {base_mape:.2f}% to {current_mape:.2f}%")
                important_regressors[regressor] = prior_scale
                base_mape = current_mape
            else:
                logger.debug(f"Regressor {regressor} does not improve performance (MAPE: {current_mape:.2f}%)")
            
            return important_regressors, base_mape
            
        except Exception as e:
            logger.warning(f"Error testing regressor {regressor}: {str(e)}")
            return important_regressors, base_mape