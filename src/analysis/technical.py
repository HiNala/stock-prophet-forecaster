"""
Technical analysis module with advanced indicators and caching
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
import functools
from concurrent.futures import ThreadPoolExecutor
from ..utils import get_logger, get_config

logger = get_logger()
config = get_config()

def cache_result(func):
    """Decorator for caching function results"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        cache_key = f"{func.__name__}_{hash(str(args))}_{hash(str(kwargs))}"
        return self._get_cached_data(cache_key, lambda: func(self, *args, **kwargs))
    return wrapper

class TechnicalAnalyzer:
    def __init__(self):
        """Initialize technical analyzer with caching and parallel processing"""
        self._cache = {}
        self._max_cache_size = 100
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        
    def cleanup(self):
        """Clean up resources"""
        self._thread_pool.shutdown(wait=False)
        self._cache.clear()
        
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
    def calculate_indicators(self, data: pd.DataFrame, 
                           selected_indicators: Optional[Dict[str, bool]] = None) -> pd.DataFrame:
        """Calculate technical indicators with parallel processing"""
        try:
            df = data.copy()
            
            # Determine which indicators to calculate
            indicators_to_calculate = {
                'momentum': self._calculate_momentum_indicators,
                'trend': self._calculate_trend_indicators,
                'volatility': self._calculate_volatility_indicators,
                'volume': self._calculate_volume_indicators
            }
            
            # Calculate selected indicators in parallel
            futures = []
            for indicator_type, calc_func in indicators_to_calculate.items():
                if (not selected_indicators or 
                    any(ind in selected_indicators for ind in self._get_indicator_names(indicator_type))):
                    futures.append(
                        self._thread_pool.submit(calc_func, df)
                    )
            
            # Collect results
            for future in futures:
                result = future.result()
                if result is not None:
                    df = pd.concat([df, result], axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise
            
    def _get_indicator_names(self, indicator_type: str) -> List[str]:
        """Get indicator names by type"""
        indicators = {
            'momentum': ['rsi', 'macd', 'stoch'],
            'trend': ['sma', 'ema'],
            'volatility': ['bollinger', 'atr'],
            'volume': ['obv', 'vwap']
        }
        return indicators.get(indicator_type, [])
        
    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators efficiently"""
        try:
            df = pd.DataFrame(index=data.index)
            
            # RSI
            df['rsi'] = RSIIndicator(data['close']).rsi()
            
            # MACD
            macd = MACD(data['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # Stochastic
            stoch = StochasticOscillator(data['high'], data['low'], data['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {str(e)}")
            return None
            
    def _calculate_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend indicators efficiently"""
        try:
            df = pd.DataFrame(index=data.index)
            
            # Moving averages
            for period in [20, 50, 200]:
                df[f'sma_{period}'] = SMAIndicator(
                    data['close'], window=period
                ).sma_indicator()
                df[f'ema_{period}'] = EMAIndicator(
                    data['close'], window=period
                ).ema_indicator()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {str(e)}")
            return None
            
    def _calculate_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators efficiently"""
        try:
            df = pd.DataFrame(index=data.index)
            
            # Bollinger Bands
            bb = BollingerBands(data['close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            
            # ATR
            df['atr'] = AverageTrueRange(
                data['high'], data['low'], data['close']
            ).average_true_range()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {str(e)}")
            return None
            
    def _calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume indicators efficiently"""
        try:
            df = pd.DataFrame(index=data.index)
            
            # On Balance Volume
            df['obv'] = OnBalanceVolumeIndicator(
                data['close'], data['volume']
            ).on_balance_volume()
            
            # Volume Weighted Average Price
            df['vwap'] = VolumeWeightedAveragePrice(
                data['high'], data['low'], data['close'], data['volume']
            ).volume_weighted_average_price()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {str(e)}")
            return None
            
    def analyze_trend(self, data: pd.DataFrame) -> Dict:
        """Analyze overall trend and strength"""
        try:
            # Calculate basic trend metrics
            returns = data['close'].pct_change()
            volatility = returns.std()
            trend = (data['close'].iloc[-1] / data['close'].iloc[0]) - 1
            
            # Calculate trend strength using moving averages
            sma_20 = SMAIndicator(data['close'], window=20).sma_indicator()
            sma_50 = SMAIndicator(data['close'], window=50).sma_indicator()
            
            # Determine trend direction and strength
            trend_direction = 'Upward' if trend > 0 else 'Downward' if trend < 0 else 'Sideways'
            ma_trend = 'Bullish' if sma_20.iloc[-1] > sma_50.iloc[-1] else 'Bearish'
            
            return {
                'direction': trend_direction,
                'strength': abs(trend),
                'ma_trend': ma_trend,
                'volatility': volatility,
                'return': trend * 100,  # as percentage
                'risk_level': 'High' if volatility > 0.02 else 'Medium' if volatility > 0.01 else 'Low'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {str(e)}")
            return {} 