"""
Enhanced data processing module with outlier detection and robust error handling.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List, Union
import yfinance as yf
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor
import functools
from scipy import stats
import logging
from contextlib import contextmanager
from ..utils.error_handler import StockProphetError

# Initialize basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessingError(Exception):
    """Raised when data processing operations fail."""
    pass

class DataFetchError(Exception):
    """Raised when data fetching operations fail."""
    pass

# Performance monitoring
class PerformanceMetrics:
    def __init__(self):
        self.metrics = {}
    
    def record_metric(self, name: str, value: float):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_metric_stats(self, name: str) -> Optional[Dict[str, float]]:
        if name not in self.metrics or not self.metrics[name]:
            return None
        values = self.metrics[name]
        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "count": len(values)
        }

monitor = PerformanceMetrics()

@contextmanager
def Timer(description: str = "Operation"):
    """Context manager for timing operations."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logger.info(f"{description} completed in {elapsed_time:.2f} seconds")
        monitor.record_metric(f"time_{description.lower().replace(' ', '_')}", elapsed_time)

def log_execution_time(func):
    """Decorator for logging function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {elapsed_time:.2f} seconds")
        monitor.record_metric(f"time_{func.__name__}", elapsed_time)
        return result
    return wrapper

# Default configuration
DEFAULT_CONFIG = {
    "DATA": {
        "min_periods": 20,
        "outlier_std_threshold": 3,
        "min_data_points": 30,
        "max_data_points": 2000,
        "max_missing_percentage": 0.1,
        "default_history_days": 365,
        "cache_size": 100,
        "cache_validity_minutes": 5,
        "max_workers": 4,
        "retry_delay_seconds": 1,
        "min_price": 0.01,
        "max_price": 1000000,
        "required_columns": ["open", "high", "low", "close", "volume"]
    },
    "TECHNICAL": {
        "sma_periods": [20, 50, 200],
        "ema_periods": [12, 26],
        "rsi_period": 14,
        "macd_periods": {"fast": 12, "slow": 26, "signal": 9},
        "volatility_window": 20,
        "bollinger_window": 20,
        "bollinger_std": 2
    }
}

def validate_required_columns(df, required_columns):
    """Validate that DataFrame contains required columns."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise DataProcessingError(f"Missing required columns: {missing_columns}")

def validate_data_points(df, min_points, max_points):
    """Validate that DataFrame has required number of data points."""
    num_points = len(df)
    if num_points < min_points:
        raise DataProcessingError(f"Insufficient data points: {num_points} < {min_points}")
    if num_points > max_points:
        raise DataProcessingError(f"Too many data points: {num_points} > {max_points}")

def cache_result(func):
    """Decorator for caching function results with error handling."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            cache_key = f"{func.__name__}_{hash(str(args))}_{hash(str(kwargs))}"
            return self._get_cached_data(cache_key, lambda: func(self, *args, **kwargs))
        except Exception as e:
            logger.error(f"Cache error in {func.__name__}: {str(e)}", exc_info=True)
            return func(self, *args, **kwargs)  # Fallback to direct execution
    return wrapper

class DataProcessor:
    def __init__(self, config=None):
        """Initialize DataProcessor with configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        # Default technical indicator configuration
        default_config = {
            "technical_indicators": {
                "sma_periods": [20, 50, 200],
                "rsi_period": 14,
                "macd_periods": {
                    "fast": 12,
                    "slow": 26,
                    "signal": 9
                },
                "bollinger_window": 20,
                "bollinger_std": 2,
                "volatility_window": 20
            }
        }
        
        self.config = config if config is not None else default_config
        self._cache = {}
        self.logger = logging.getLogger(__name__)
        
    @log_execution_time
    def cleanup(self):
        """Clean up resources with proper error handling."""
        try:
            if self._thread_pool:
                self._thread_pool.shutdown(wait=False)
            self._cache.clear()
            self._last_fetch.clear()
            logger.info("Successfully cleaned up DataProcessor resources")
        except Exception as e:
            logger.error("Error during cleanup", exc_info=True)
            raise DataProcessingError("Failed to clean up resources") from e
        
    @log_execution_time
    def _clear_cache(self):
        """Clear cache with size monitoring."""
        try:
            if len(self._cache) > self._max_cache_size:
                oldest_keys = sorted(self._cache.keys())[:(len(self._cache) - self._max_cache_size)]
                for key in oldest_keys:
                    self._cache.pop(key, None)
                logger.debug(f"Cleared {len(oldest_keys)} old cache entries")
                monitor.record_metric("cache_size", len(self._cache))
        except Exception as e:
            logger.error("Error clearing cache", exc_info=True)
            self._cache.clear()
            raise DataProcessingError("Failed to clear cache") from e
        
    @log_execution_time
    def _get_cached_data(self, key: str, data_fn) -> pd.DataFrame:
        """Get data from cache with validation."""
        try:
            if key not in self._cache:
                with Timer(f"Computing data for {key}"):
                    self._cache[key] = data_fn()
                    self._clear_cache()
                    monitor.record_metric("cache_misses", 1)
            else:
                monitor.record_metric("cache_hits", 1)
            return self._cache[key]
        except Exception as e:
            logger.error("Error accessing cache", exc_info=True)
            return data_fn()
        
    @log_execution_time
    def detect_outliers(self, data: pd.DataFrame, columns: List[str] = None,
                       method: str = "zscore", threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect and handle outliers in specified columns.
        
        Args:
            data: Input DataFrame
            columns: List of columns to check for outliers
            method: Detection method ('zscore' or 'iqr')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outlier information
        """
        try:
            if columns is None:
                columns = DEFAULT_CONFIG["DATA"]["required_columns"]
            
            outliers_info = pd.DataFrame(index=data.index)
            
            for column in columns:
                if column not in data.columns:
                    continue
                    
                with Timer(f"Detecting outliers in {column}"):
                    values = data[column].values
                    if method == "zscore":
                        z_scores = np.abs(stats.zscore(values))
                        outliers = z_scores > threshold
                    else:  # IQR method
                        Q1 = np.percentile(values, 25)
                        Q3 = np.percentile(values, 75)
                        IQR = Q3 - Q1
                        outliers = (values < (Q1 - threshold * IQR)) | (values > (Q3 + threshold * IQR))
                    
                    outliers_info[f"{column}_outlier"] = outliers
                    if outliers.any():
                        logger.warning(f"Detected {outliers.sum()} outliers in {column}")
                        monitor.record_metric(f"outliers_{column}", outliers.sum())
                    
            return outliers_info
            
        except Exception as e:
            logger.error("Error detecting outliers", exc_info=True)
            raise DataProcessingError("Failed to detect outliers") from e
    
    @log_execution_time
    def handle_outliers(self, data: pd.DataFrame, outliers_info: pd.DataFrame,
                       method: str = "clip") -> pd.DataFrame:
        """
        Handle detected outliers using specified method.
        
        Args:
            data: Input DataFrame
            outliers_info: DataFrame with outlier information
            method: Handling method ('clip', 'remove', or 'interpolate')
            
        Returns:
            DataFrame with handled outliers
        """
        try:
            df = data.copy()
            
            for column in data.columns:
                outlier_col = f"{column}_outlier"
                if outlier_col not in outliers_info.columns:
                    continue
                    
                outliers = outliers_info[outlier_col]
                if not outliers.any():
                    continue
                    
                with Timer(f"Handling outliers in {column}"):
                    if method == "clip":
                        values = df[column].values
                        Q1 = np.percentile(values, 25)
                        Q3 = np.percentile(values, 75)
                        IQR = Q3 - Q1
                        df.loc[outliers, column] = np.clip(
                            df.loc[outliers, column],
                            Q1 - 1.5 * IQR,
                            Q3 + 1.5 * IQR
                        ).astype(df[column].dtype)
                    elif method == "remove":
                        df = df[~outliers]
                    else:  # interpolate
                        df.loc[outliers, column] = np.nan
                        df[column] = df[column].interpolate(method='linear')
                    
                    monitor.record_metric(f"handled_outliers_{column}", outliers.sum())
                    
            return df
            
        except Exception as e:
            logger.error("Error handling outliers", exc_info=True)
            raise DataProcessingError("Failed to handle outliers") from e
        
    @cache_result
    @log_execution_time
    def fetch_data(self, symbol: str, start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Fetch stock data with enhanced error handling and validation."""
        try:
            # Validate inputs
            if not symbol or not isinstance(symbol, str):
                raise DataValidationError("Invalid symbol provided")
            
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.now()
            
            if not start_date:
                start_date = end_date - timedelta(days=180)  # Default to 6 months
            
            # Ensure dates are timezone-naive
            if start_date.tzinfo:
                start_date = start_date.replace(tzinfo=None)
            if end_date.tzinfo:
                end_date = end_date.replace(tzinfo=None)
            
            # Calculate the actual time difference
            days_diff = (end_date - start_date).days
            logger.debug(f"Requested time period: {days_diff} days")
            
            # Add a small buffer to the end date to ensure we get the latest data
            end_date = end_date + timedelta(days=1)
            
            # Handle multiple tickers
            symbols = [s.strip().upper() for s in symbol.split(',')]
            logger.debug(f"Processing symbols: {symbols}")
            
            # Check cache first
            cache_key = f"{symbol}_{start_date.date()}_{end_date.date()}"
            if cache_key in self._cache:
                logger.debug(f"Using cached data for {symbol}")
                monitor.record_metric("cache_hits", 1)
                return self._cache[cache_key]
            
            monitor.record_metric("cache_misses", 1)
            
            # Use yfinance download for multiple symbols
            for attempt in range(3):
                try:
                    with Timer(f"Downloading data for {symbol} (attempt {attempt + 1})"):
                        data = yf.download(
                            tickers=symbols,
                            start=start_date,
                            end=end_date,
                            interval='1d',
                            auto_adjust=True,
                            actions=True,
                            group_by='ticker',
                            threads=True,
                            progress=False
                        )
                        
                        if data.empty:
                            raise DataFetchError(f"No data found for symbols: {symbols}")
                        
                        logger.debug(f"Downloaded data shape: {data.shape}")
                        monitor.record_metric("downloaded_rows", len(data))
                        
                        # Process data based on number of symbols
                        if len(symbols) == 1:
                            return self._process_single_symbol(data, symbol, days_diff, cache_key)
                        else:
                            return self._process_multiple_symbols(data, symbols, cache_key)
                            
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        raise DataFetchError(f"Failed to fetch data after 3 attempts: {str(e)}")
                    logger.warning(f"Retry {attempt + 1} for {symbol}: {str(e)}")
                    time.sleep(self._retry_delay * (attempt + 1))
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}", exc_info=True)
            raise DataFetchError(f"Failed to fetch data: {str(e)}") from e
        
    @log_execution_time
    def _process_single_symbol(self, data: pd.DataFrame, symbol: str, days_diff: int, cache_key: str) -> pd.DataFrame:
        """Process data for a single symbol."""
        try:
            with Timer(f"Processing single symbol {symbol}"):
                processed_data = self._validate_and_clean_data(data)
                if processed_data is None or processed_data.empty:
                    raise DataProcessingError(f"No valid data after processing for {symbol}")
                
                logger.debug(f"Processed data shape: {processed_data.shape}")
                actual_days = (processed_data.index.max() - processed_data.index.min()).days
                logger.debug(f"Actual days in data: {actual_days}")
                
                # Verify data completeness
                if actual_days < days_diff * 0.9:  # Allow for 10% missing days
                    logger.warning(f"Received less data than requested: {actual_days} vs {days_diff} days")
                    monitor.record_metric("incomplete_data_ratio", actual_days / days_diff)
                
                # Detect and handle outliers
                outliers_info = self.detect_outliers(processed_data)
                if not outliers_info.empty:
                    processed_data = self.handle_outliers(processed_data, outliers_info)
                
                self._cache[cache_key] = processed_data
                return processed_data
                
        except Exception as e:
            logger.error(f"Error processing single symbol {symbol}", exc_info=True)
            raise DataProcessingError(f"Failed to process data for {symbol}") from e

    @log_execution_time
    def _process_multiple_symbols(self, data: pd.DataFrame, symbols: List[str], cache_key: str) -> Dict[str, pd.DataFrame]:
        """Process data for multiple symbols."""
        try:
            results = {}
            with Timer(f"Processing multiple symbols: {', '.join(symbols)}"):
                for sym in symbols:
                    if sym in data.columns.levels[0]:
                        with Timer(f"Processing symbol {sym}"):
                            ticker_data = data[sym].copy()
                            processed_data = self._validate_and_clean_data(ticker_data)
                            if processed_data is not None and not processed_data.empty:
                                outliers_info = self.detect_outliers(processed_data)
                                if not outliers_info.empty:
                                    processed_data = self.handle_outliers(processed_data, outliers_info)
                                results[sym] = processed_data
                                monitor.record_metric(f"processed_rows_{sym}", len(processed_data))
                            else:
                                logger.warning(f"No valid data after processing for {sym}")
                    else:
                        logger.warning(f"No data found for {sym}")
                
                if not results:
                    raise DataProcessingError(f"No valid data found for any symbols in: {symbols}")
                
                self._cache[cache_key] = results
                return results
                
        except Exception as e:
            logger.error("Error processing multiple symbols", exc_info=True)
            raise DataProcessingError("Failed to process multiple symbols") from e

    @log_execution_time
    def _validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean stock data with enhanced preprocessing."""
        try:
            if data is None:
                raise DataValidationError("Input data is None")
            
            if not isinstance(data, pd.DataFrame):
                raise DataValidationError("Input must be a pandas DataFrame")
            
            with Timer("Data validation and cleaning"):
                # Make a copy to avoid modifying the original
                df = data.copy()
                
                # Handle MultiIndex columns (from multiple tickers)
                if isinstance(df.columns, pd.MultiIndex):
                    logger.debug("Detected MultiIndex columns from yfinance")
                    ticker = df.columns.levels[0][0]
                    logger.debug(f"Processing ticker: {ticker}")
                    
                    # Create a new DataFrame with flattened column names
                    new_df = pd.DataFrame(index=df.index)
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']:
                        if (ticker, col) in df.columns:
                            new_df[col.lower()] = df[(ticker, col)]
                    df = new_df
                
                # Ensure datetime index
                if not isinstance(df.index, pd.DatetimeIndex):
                    logger.warning("Converting index to datetime")
                    df.index = pd.to_datetime(df.index)
                
                # Remove timezone info for consistency
                df.index = df.index.tz_localize(None)
                
                # Convert all column names to lowercase
                df.columns = df.columns.str.lower()
                
                # Validate required columns
                required_columns = DEFAULT_CONFIG["DATA"]["required_columns"]
                validate_required_columns(df, required_columns)
                
                # Validate data points
                validate_data_points(df, DEFAULT_CONFIG["DATA"]["min_data_points"], DEFAULT_CONFIG["DATA"]["max_data_points"])
                
                # Store original values before any modifications
                for col in required_columns:
                    df[f'{col}_original'] = df[col].copy()
                
                # Validate price ranges
                with Timer("Validating price ranges"):
                    self._validate_price_ranges(df)
                
                # Add technical indicators
                with Timer("Adding technical indicators"):
                    df = self._add_technical_indicators(df)
                
                monitor.record_metric("processed_rows", len(df))
                return df
                
        except Exception as e:
            logger.error("Error in data validation", exc_info=True)
            raise DataProcessingError("Failed to validate and clean data") from e

    def _validate_price_ranges(self, df: pd.DataFrame) -> None:
        """Validate and fix price ranges."""
        try:
            min_price = DEFAULT_CONFIG["DATA"]["min_price"]
            max_price = DEFAULT_CONFIG["DATA"]["max_price"]
            
            for col in ['open', 'high', 'low', 'close']:
                invalid_prices = (df[col] < min_price) | (df[col] > max_price)
                if invalid_prices.any():
                    logger.warning(f"Found {invalid_prices.sum()} invalid prices in {col}")
                    monitor.record_metric(f"invalid_prices_{col}", invalid_prices.sum())
                    df.loc[invalid_prices, col] = df.loc[invalid_prices, f'{col}_original'].rolling(
                        window=3, min_periods=1, center=True
                    ).mean()
            
            # Validate high/low relationship
            invalid_hl = df['high'] < df['low']
            if invalid_hl.any():
                logger.warning(f"Found {invalid_hl.sum()} invalid high/low relationships")
                monitor.record_metric("invalid_high_low", invalid_hl.sum())
                df.loc[invalid_hl, ['high', 'low']] = df.loc[invalid_hl, ['low', 'high']].values
            
            # Handle volume data
            df['volume'] = df['volume'].replace(0, np.nan)
            df['volume'] = df['volume'].fillna(df['volume'].rolling(window=5, min_periods=1).mean())
            
        except Exception as e:
            logger.error("Error validating price ranges", exc_info=True)
            raise DataProcessingError("Failed to validate price ranges") from e

    @log_execution_time
    def calculate_indicators(self, df: pd.DataFrame, selected_indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Public interface to calculate technical indicators.
        
        Args:
            df: DataFrame containing the stock data
            selected_indicators: Optional list of specific indicators to calculate. If None, calculates all.
            
        Returns:
            DataFrame with technical indicators added
        """
        return self._add_technical_indicators(df)

    @log_execution_time
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe."""
        try:
            tech_config = self.config.get("technical_indicators", {})
            if not tech_config:
                self.logger.warning("Using default technical indicator settings")
                tech_config = self.config["technical_indicators"]
            
            with Timer("Calculating moving averages"):
                # Enhanced moving averages
                df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
                df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()
                df['sma_200'] = df['close'].rolling(window=200, min_periods=1).mean()
                df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
                df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
                
                # Add lagged returns for momentum
                df['returns_1d'] = df['close'].pct_change(1)
                df['returns_5d'] = df['close'].pct_change(5)
                df['returns_20d'] = df['close'].pct_change(20)
                
                monitor.record_metric("ma_calc_time", time.perf_counter())
            
            with Timer("Calculating RSI"):
                # RSI with smoothing
                rsi_period = tech_config.get("rsi_period", 14)
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                df['rsi_sma'] = df['rsi'].rolling(window=5).mean()  # Smoothed RSI
                monitor.record_metric("rsi_calc_time", time.perf_counter())
            
            with Timer("Calculating MACD"):
                # Enhanced MACD
                macd_config = tech_config.get("macd_periods", {})
                fast_period = macd_config.get("fast", 12)
                slow_period = macd_config.get("slow", 26)
                signal_period = macd_config.get("signal", 9)
                
                ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
                ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
                df['macd'] = ema_fast - ema_slow
                df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
                df['macd_hist'] = df['macd'] - df['macd_signal']
                df['macd_divergence'] = df['macd'].diff()
                monitor.record_metric("macd_calc_time", time.perf_counter())
            
            with Timer("Calculating Bollinger Bands"):
                # Enhanced Bollinger Bands
                bb_window = tech_config.get("bollinger_window", 20)
                bb_std = tech_config.get("bollinger_std", 2)
                rolling_mean = df['close'].rolling(window=bb_window).mean()
                rolling_std = df['close'].rolling(window=bb_window).std()
                df['bb_upper'] = rolling_mean + (rolling_std * bb_std)
                df['bb_lower'] = rolling_mean - (rolling_std * bb_std)
                df['bb_middle'] = rolling_mean
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
                monitor.record_metric("bb_calc_time", time.perf_counter())
            
            with Timer("Calculating returns and volatility"):
                # Enhanced volatility metrics
                df['returns'] = df['close'].pct_change()
                df['volatility'] = df['returns'].rolling(
                    window=tech_config.get("volatility_window", 20),
                    min_periods=1
                ).std()
                df['volatility_sma'] = df['volatility'].rolling(window=7).mean()
                df['volatility_trend'] = df['volatility_sma'].diff()
                monitor.record_metric("volatility_calc_time", time.perf_counter())
            
            # Fill NaN values
            df = df.ffill().bfill()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {str(e)}", exc_info=True)
            raise DataProcessingError("Failed to add technical indicators") from e

    def smooth_series(self, series: pd.Series, window: int = 7) -> pd.Series:
        """Apply smoothing to a time series using rolling average.
        
        Args:
            series: Input time series data
            window: Rolling window size
            
        Returns:
            Smoothed time series
        """
        return series.rolling(window=window, min_periods=1, center=True).mean()

    def select_important_features(self, df: pd.DataFrame, threshold: float = 0.1) -> List[str]:
        """Dynamically select important features based on their predictive power.
        
        Args:
            df: Input dataframe with features
            threshold: Minimum variance threshold for feature selection
            
        Returns:
            List of selected feature names
        """
        selected_features = []
        technical_features = ['trend_consistency', 'volatility', 'price_momentum', 'volume_trend']
        
        for feature in technical_features:
            if feature in df.columns:
                # Check feature variance
                feature_std = df[feature].std()
                if feature_std > threshold:
                    # Apply smoothing to reduce noise
                    df[feature] = self.smooth_series(df[feature])
                    selected_features.append(feature)
                else:
                    self.logger.warning(f"Feature {feature} has low variance ({feature_std:.4f}), excluding from model")
        
        return selected_features

    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features by handling outliers and scaling.
        
        Args:
            df: Input dataframe
        
        Returns:
            Preprocessed dataframe
        """
        processed_df = df.copy()
        
        # Handle outliers in volume using quantile-based clipping
        volume_q1, volume_q99 = processed_df['volume'].quantile([0.01, 0.99])
        processed_df['volume'] = np.clip(
            processed_df['volume'], 
            volume_q1, 
            volume_q99
        )
        
        # Smooth technical indicators
        technical_features = [
            'trend_consistency', 'volatility', 
            'price_momentum', 'volume_trend'
        ]
        
        for feature in technical_features:
            if feature in processed_df.columns:
                # Apply smoothing
                processed_df[feature] = self.smooth_series(
                    processed_df[feature], 
                    window=7
                )
                
                # Normalize feature
                feature_mean = processed_df[feature].mean()
                feature_std = processed_df[feature].std()
                if feature_std > 0:
                    processed_df[feature] = (
                        processed_df[feature] - feature_mean
                    ) / feature_std
        
        return processed_df

    def validate_features(self, df: pd.DataFrame, 
                         min_variance: float = 0.01,
                         correlation_threshold: float = 0.7) -> List[str]:
        """Validate features based on variance, correlation, and predictive power."""
        features = [
            'trend_consistency', 'volatility', 'volatility_sma',
            'price_momentum', 'volume_trend', 'returns_5d',
            'returns_20d', 'rsi_sma', 'macd_divergence',
            'bb_width', 'volatility_trend'
        ]
        
        validated_features = []
        feature_scores = {}
        
        # Check variance and calculate feature scores
        for feature in features:
            if feature not in df.columns:
                continue
            
            variance = df[feature].var()
            if variance >= min_variance:
                # Calculate feature score based on correlation with target
                correlation = abs(df[feature].corr(df['y']))
                feature_scores[feature] = correlation
                validated_features.append(feature)
            else:
                self.logger.warning(
                    f"Feature {feature} has low variance ({variance:.4f})"
                )
        
        # Sort features by correlation score
        validated_features.sort(key=lambda x: feature_scores[x], reverse=True)
        
        # Keep only top performing uncorrelated features
        final_features = []
        for feature in validated_features:
            # Check correlation with already selected features
            if not final_features or all(
                abs(df[feature].corr(df[f])) < correlation_threshold 
                for f in final_features
            ):
                final_features.append(feature)
                if len(final_features) >= 4:  # Limit to top 4 features
                    break
        
        return final_features