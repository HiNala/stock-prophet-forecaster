import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
from datetime import datetime, timedelta

def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate percentage returns"""
    return prices.pct_change()

def calculate_volatility(returns: pd.Series, window: int = 252) -> pd.Series:
    """Calculate rolling volatility"""
    return returns.rolling(window=window).std() * np.sqrt(window)

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio"""
    excess_returns = returns - risk_free_rate/252
    if len(excess_returns) == 0:
        return 0
    return np.sqrt(252) * excess_returns.mean() / returns.std()

def calculate_drawdown(prices: pd.Series) -> Tuple[pd.Series, float, int]:
    """Calculate drawdown series and maximum drawdown"""
    rolling_max = prices.expanding(min_periods=1).max()
    drawdown = (prices - rolling_max) / rolling_max
    
    max_drawdown = drawdown.min()
    max_drawdown_duration = get_max_drawdown_duration(drawdown)
    
    return drawdown, max_drawdown, max_drawdown_duration

def get_max_drawdown_duration(drawdown: pd.Series) -> int:
    """Calculate maximum drawdown duration in days"""
    is_drawdown = drawdown < 0
    
    # Find the end of each drawdown period
    drawdown_end = pd.Series(index=drawdown.index, data=False)
    drawdown_end[1:] = is_drawdown[:-1] & ~is_drawdown[1:]
    
    # Find the start of each drawdown period
    drawdown_start = pd.Series(index=drawdown.index, data=False)
    drawdown_start[:-1] = ~is_drawdown[:-1] & is_drawdown[1:]
    
    # Calculate durations
    durations = []
    current_start = None
    
    for i, (start, end) in enumerate(zip(drawdown_start, drawdown_end)):
        if start:
            current_start = drawdown.index[i]
        elif end and current_start is not None:
            duration = (drawdown.index[i] - current_start).days
            durations.append(duration)
            current_start = None
            
    return max(durations) if durations else 0

def format_number(num: float, precision: int = 2) -> str:
    """Format numbers for display"""
    if abs(num) >= 1e9:
        return f"{num/1e9:.{precision}f}B"
    elif abs(num) >= 1e6:
        return f"{num/1e6:.{precision}f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"

def calculate_position_size(capital: float,
                          risk_per_trade: float,
                          stop_loss_pct: float,
                          entry_price: float) -> Tuple[int, float]:
    """
    Calculate position size based on risk management rules
    
    Args:
        capital: Available capital
        risk_per_trade: Maximum risk per trade as percentage of capital
        stop_loss_pct: Stop loss percentage
        entry_price: Entry price of the asset
        
    Returns:
        Tuple of (quantity, dollar_risk)
    """
    max_risk_amount = capital * risk_per_trade
    price_risk = entry_price * stop_loss_pct
    
    # Calculate quantity based on risk
    quantity = int(max_risk_amount / price_risk)
    
    # Calculate actual dollar risk
    dollar_risk = quantity * price_risk
    
    return quantity, dollar_risk

def validate_price_data(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validate price data for analysis
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Check required columns
    if not all(col in df.columns for col in required_columns):
        return False, "Missing required columns"
        
    # Check for missing values
    if df[required_columns].isnull().any().any():
        return False, "Data contains missing values"
        
    # Check for negative prices
    if (df[['Open', 'High', 'Low', 'Close']] <= 0).any().any():
        return False, "Data contains invalid negative prices"
        
    # Check for price anomalies
    if not ((df['High'] >= df['Low']).all() and
            (df['High'] >= df['Open']).all() and
            (df['High'] >= df['Close']).all() and
            (df['Low'] <= df['Open']).all() and
            (df['Low'] <= df['Close']).all()):
        return False, "Data contains price anomalies"
        
    return True, None

def get_market_hours(date: datetime) -> Tuple[datetime, datetime]:
    """Get market opening and closing hours for a given date"""
    # Default to US market hours (9:30 AM - 4:00 PM EST)
    market_open = datetime.combine(date.date(), datetime.strptime("09:30", "%H:%M").time())
    market_close = datetime.combine(date.date(), datetime.strptime("16:00", "%H:%M").time())
    
    return market_open, market_close

def is_market_open(dt: datetime = None) -> bool:
    """Check if market is currently open"""
    if dt is None:
        dt = datetime.now()
        
    # Check if weekend
    if dt.weekday() in [5, 6]:  # Saturday = 5, Sunday = 6
        return False
        
    market_open, market_close = get_market_hours(dt)
    
    return market_open <= dt <= market_close 