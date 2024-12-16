"""
Enhanced data processing module for stock data
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
import yfinance as yf
from datetime import datetime, timedelta
import time
import ta
from concurrent.futures import ThreadPoolExecutor
import functools
from ..utils import get_logger, get_config

logger = get_logger()
config = get_config()

def cache_result(func):
    """Decorator for caching function results"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            cache_key = f"{func.__name__}_{hash(str(args))}_{hash(str(kwargs))}"
            return self._get_cached_data(cache_key, lambda: func(self, *args, **kwargs))
        except Exception as e:
            logger.error(f"Cache error in {func.__name__}: {str(e)}")
            return func(self, *args, **kwargs)  # Fallback to direct execution
    return wrapper

class DataProcessor:
    def __init__(self):
        """Initialize data processor with caching"""
        self._cache = {}
        self._max_cache_size = 100
        self._last_fetch = {}
        self._data_validity = timedelta(minutes=5)
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        self._retry_delay = 1  # Delay between retries in seconds
        
    def cleanup(self):
        """Clean up resources"""
        try:
            if self._thread_pool:
                self._thread_pool.shutdown(wait=False)
            self._cache.clear()
            self._last_fetch.clear()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
        
    def _clear_cache(self):
        """Clear cache if it gets too large"""
        try:
            if len(self._cache) > self._max_cache_size:
                oldest_keys = sorted(self._cache.keys())[:(len(self._cache) - self._max_cache_size)]
                for key in oldest_keys:
                    self._cache.pop(key, None)
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            self._cache.clear()  # Fallback to clearing entire cache
            
    def _get_cached_data(self, key, data_fn):
        """Get data from cache or compute it"""
        try:
            if key not in self._cache:
                self._cache[key] = data_fn()
                self._clear_cache()
            return self._cache[key]
        except Exception as e:
            logger.error(f"Error accessing cache: {str(e)}")
            return data_fn()  # Fallback to direct computation
        
    @cache_result
    def fetch_data(self, symbol: str, start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Fetch stock data with improved error handling and validation"""
        try:
            # Validate inputs
            if not symbol or not isinstance(symbol, str):
                raise ValueError("Invalid symbol provided")
            
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
            logger.debug(f"Date range: {start_date.date()} to {end_date.date()}")
            
            # Check cache first
            cache_key = f"{symbol}_{start_date.date()}_{end_date.date()}"
            if cache_key in self._cache:
                logger.debug(f"Using cached data for {symbol}")
                return self._cache[cache_key]
            
            # Use yfinance download for multiple symbols
            for attempt in range(3):
                try:
                    logger.debug(f"Downloading data (attempt {attempt + 1}/3)")
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
                        raise ValueError(f"No data found for symbols: {symbols}")
                    
                    logger.debug(f"Downloaded data shape: {data.shape}")
                    if isinstance(data.columns, pd.MultiIndex):
                        logger.debug(f"Column levels: {data.columns.levels}")
                    else:
                        logger.debug(f"Columns: {data.columns.tolist()}")
                    
                    # Handle single vs multiple ticker response
                    if len(symbols) == 1:
                        data = self._validate_and_clean_data(data)
                        if data is not None and not data.empty:
                            logger.debug(f"Processed data shape: {data.shape}")
                            logger.debug(f"Date range in data: {data.index.min()} to {data.index.max()}")
                            actual_days = (data.index.max() - data.index.min()).days
                            logger.debug(f"Actual days in data: {actual_days}")
                            
                            # Verify data completeness
                            if actual_days < days_diff * 0.9:  # Allow for 10% missing days
                                logger.warning(f"Received less data than requested: {actual_days} vs {days_diff} days")
                                # Try to fetch with a larger date range to compensate
                                if attempt < 2:
                                    logger.debug("Retrying with extended date range")
                                    start_date = start_date - timedelta(days=int(days_diff * 0.2))  # Add 20% buffer
                                    continue
                            
                            self._cache[cache_key] = data
                            return data
                        else:
                            raise ValueError(f"No valid data after processing for {symbols[0]}")
                    else:
                        # Process each ticker's data
                        results = {}
                        for sym in symbols:
                            if sym in data.columns.levels[0]:  # Check if we have data for this symbol
                                ticker_data = data[sym].copy()
                                processed_data = self._validate_and_clean_data(ticker_data)
                                if processed_data is not None and not processed_data.empty:
                                    results[sym] = processed_data
                                else:
                                    logger.warning(f"No valid data after processing for {sym}")
                            else:
                                logger.warning(f"No data found for {sym}")
                        
                        if not results:
                            raise ValueError(f"No valid data found for any symbols in: {symbols}")
                        
                        self._cache[cache_key] = results
                        return results
                    
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        raise
                    logger.warning(f"Retry {attempt + 1} for {symbol}: {str(e)}")
                    time.sleep(self._retry_delay * (attempt + 1))
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
            
    def _validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean stock data with enhanced preprocessing"""
        try:
            if data is None:
                raise ValueError("Input data is None")
            
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
            
            # Make a copy to avoid modifying the original
            df = data.copy()
            
            # Handle MultiIndex columns (from multiple tickers)
            if isinstance(df.columns, pd.MultiIndex):
                logger.debug("Detected MultiIndex columns from yfinance")
                # Get the first ticker's data
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
            
            # Check required columns (case-insensitive)
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Available columns: {df.columns.tolist()}")
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Store original values before any modifications
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[f'{col}_original'] = df[col].copy()
                logger.debug(f"Created {col}_original column with values: {df[f'{col}_original'].head()}")
            
            # Fill missing values in price columns first
            price_cols = ['open', 'high', 'low', 'close']
            df[price_cols] = df[price_cols].ffill().bfill()
            
            # Remove rows with invalid prices (zero or negative)
            price_mask = (
                (df['open'] > 0) &
                (df['high'] > 0) &
                (df['low'] > 0) &
                (df['close'] > 0) &
                (df['high'] >= df['low'])
            )
            
            invalid_prices = ~price_mask
            if invalid_prices.any():
                logger.warning(f"Removing {invalid_prices.sum()} rows with invalid prices")
                df = df[price_mask]
            
            # Calculate returns and volatility with min_periods
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20, min_periods=1).std()
            
            # Add price-based features with proper min_periods
            df['price_range'] = (df['high'] - df['low']) / df['close']
            df['price_momentum'] = df['close'].pct_change(5)
            df['price_acceleration'] = df['price_momentum'].diff()
            
            # Calculate moving averages with min_periods
            for window in [5, 10, 20, 50, 200]:
                df[f'sma_{window}'] = df['close'].rolling(window=window, min_periods=1).mean()
                df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False, min_periods=1).mean()
            
            # Handle volume data separately
            df['volume'] = df['volume'].replace(0, np.nan)  # Convert zeros to NaN
            df['volume'] = df['volume'].ffill().bfill()
            
            # Volume analysis with min_periods
            df['volume_sma'] = df['volume'].rolling(window=20, min_periods=1).mean()
            df['relative_volume'] = df['volume'] / df['volume_sma']
            df['volume_trend'] = df['volume_sma'].pct_change(20)
            
            # Add volatility bands with min_periods
            df['volatility_ma'] = df['volatility'].rolling(window=20, min_periods=1).mean()
            df['high_volatility'] = df['volatility'] > df['volatility_ma'] * 1.5
            
            # Add trend strength indicators
            df['adx'] = self._calculate_adx(df)
            df['trend_strength'] = df['adx'].rolling(window=14, min_periods=1).mean()
            
            # Add market regime features
            df['regime'] = np.where(
                (df['close'] > df['sma_200']) & (df['close'] > df['ema_50']),
                1,  # Strong uptrend
                np.where(
                    (df['close'] < df['sma_200']) & (df['close'] < df['ema_50']),
                    -1,  # Strong downtrend
                    0   # Sideways/uncertain
                )
            )
            
            # Add regime strength
            df['regime_strength'] = abs(
                (df['close'] - df['sma_200']) / df['sma_200'] +
                (df['close'] - df['ema_50']) / df['ema_50']
            ) * 100
            
            # Enhanced volatility analysis with min_periods
            df['volatility_fast'] = df['returns'].rolling(window=10, min_periods=1).std() * np.sqrt(252)
            df['volatility_slow'] = df['returns'].rolling(window=30, min_periods=1).std() * np.sqrt(252)
            df['volatility_ratio'] = df['volatility_fast'] / df['volatility_slow']
            
            # Volatility regimes with improved handling of duplicate values
            try:
                df['volatility_percentile'] = df['volatility'].rolling(window=252, min_periods=20).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1]
                )
                
                # Check for duplicate values
                unique_percentiles = df['volatility_percentile'].nunique()
                if unique_percentiles < 3:
                    logger.debug(f"Not enough unique volatility values (found {unique_percentiles}), using default categorization")
                    df['volatility_regime'] = 'medium'
                else:
                    try:
                        df['volatility_regime'] = pd.qcut(
                            df['volatility_percentile'],
                            q=3,
                            labels=['low', 'medium', 'high'],
                            duplicates='drop'
                        )
                    except Exception as e:
                        logger.warning(f"Error in volatility regime calculation: {str(e)}")
                        df['volatility_regime'] = 'medium'
            except Exception as e:
                logger.warning(f"Error calculating volatility percentiles: {str(e)}")
                df['volatility_percentile'] = 0.5
                df['volatility_regime'] = 'medium'
            
            # Ensure volatility_regime is string type
            df['volatility_regime'] = df['volatility_regime'].astype(str)
            
            # Add momentum indicators with min_periods
            df['momentum_1m'] = df['close'].pct_change(21)
            df['momentum_3m'] = df['close'].pct_change(63)
            df['momentum_6m'] = df['close'].pct_change(126)
            
            # Trend consistency
            df['trend_consistency'] = (
                (df['close'] > df['sma_20']).astype(float) +
                (df['close'] > df['sma_50']).astype(float) +
                (df['close'] > df['sma_200']).astype(float)
            ) / 3
            
            # Volume profile
            df['volume_profile'] = pd.qcut(
                df['volume'].rolling(window=20, min_periods=1).mean(),
                q=5,
                labels=['very_low', 'low', 'normal', 'high', 'very_high']
            ).astype(str)
            
            # Add date-based features
            df['year'] = df.index.year
            df['month'] = df.index.month
            df['day'] = df.index.day
            df['dayofweek'] = df.index.dayofweek
            df['quarter'] = df.index.quarter
            df['is_month_end'] = df.index.is_month_end.astype(int)
            df['is_month_start'] = df.index.is_month_start.astype(int)
            
            # Fill any remaining NaN values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(0)
            
            # Verify no NaN values remain
            nan_cols = df.columns[df.isna().any()].tolist()
            if nan_cols:
                logger.warning(f"Found NaN values in columns: {nan_cols}")
                for col in nan_cols:
                    df[col] = df[col].ffill().bfill().fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in data validation: {str(e)}")
            raise
            
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX)"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            # Calculate Directional Movement
            up_move = high - high.shift()
            down_move = low.shift() - low
            
            pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Calculate Directional Indicators
            pdi = 100 * pd.Series(pos_dm).rolling(window=period).mean() / atr
            ndi = 100 * pd.Series(neg_dm).rolling(window=period).mean() / atr
            
            # Calculate ADX
            dx = 100 * abs(pdi - ndi) / (pdi + ndi)
            adx = dx.rolling(window=period).mean()
            
            return adx
            
        except Exception as e:
            logger.error(f"Error calculating ADX: {str(e)}")
            return pd.Series(index=data.index)
            
    def _remove_outliers(self, data: pd.DataFrame, n_sigmas: float = 3) -> pd.DataFrame:
        """Remove outliers using IQR method with configurable threshold"""
        try:
            df = data.copy()
            
            # Calculate IQR for price columns
            for col in ['open', 'high', 'low', 'close']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - n_sigmas * IQR
                upper_bound = Q3 + n_sigmas * IQR
                
                # Count outliers
                n_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if n_outliers > 0:
                    logger.info(f"Found {n_outliers} outliers in {col}")
                
                # Replace outliers with boundary values
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
            
            return df
            
        except Exception as e:
            logger.error(f"Error removing outliers: {str(e)}")
            raise
            
    @cache_result
    def calculate_indicators(self, data: pd.DataFrame, selected_indicators: Optional[Dict[str, bool]] = None) -> pd.DataFrame:
        """Calculate technical indicators with enhanced features"""
        try:
            df = data.copy()
            
            # Default to all indicators if none selected
            if selected_indicators is None:
                selected_indicators = {
                    'sma': True, 'ema': True, 'rsi': True,
                    'macd': True, 'bollinger': True, 'stochastic': True
                }
            
            # Moving Averages
            if selected_indicators.get('sma', False):
                for period in [20, 50, 200]:
                    df[f'sma_{period}'] = df['close'].rolling(window=period, min_periods=1).mean()
            
            if selected_indicators.get('ema', False):
                for period in [20, 50, 200]:
                    df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False, min_periods=1).mean()
            
            # RSI with smoothing
            if selected_indicators.get('rsi', False):
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                # Add RSI smoothing
                df['rsi_ma'] = df['rsi'].rolling(window=5).mean()
            
            # Enhanced MACD
            if selected_indicators.get('macd', False):
                exp1 = df['close'].ewm(span=12, adjust=False).mean()
                exp2 = df['close'].ewm(span=26, adjust=False).mean()
                df['macd'] = exp1 - exp2
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                df['macd_hist'] = df['macd'] - df['macd_signal']
                # Add MACD momentum
                df['macd_momentum'] = df['macd_hist'].diff()
            
            # Bollinger Bands with dynamic multiplier
            if selected_indicators.get('bollinger', False):
                sma = df['close'].rolling(window=20).mean()
                std = df['close'].rolling(window=20).std()
                volatility = df['returns'].rolling(window=20).std()
                # Adjust band width based on volatility
                multiplier = 2 + volatility
                df['bb_upper'] = sma + (multiplier * std)
                df['bb_lower'] = sma - (multiplier * std)
                df['bb_middle'] = sma
                # Add bandwidth and %B indicators
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
                df['bb_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Enhanced Stochastic
            if selected_indicators.get('stochastic', False):
                window = 14
                df['stoch_k'] = ((df['close'] - df['low'].rolling(window=window).min()) /
                                (df['high'].rolling(window=window).max() - df['low'].rolling(window=window).min())) * 100
                df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
                # Add Stochastic RSI
                df['stoch_rsi'] = ((df['rsi'] - df['rsi'].rolling(window=window).min()) /
                                  (df['rsi'].rolling(window=window).max() - df['rsi'].rolling(window=window).min())) * 100
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise 