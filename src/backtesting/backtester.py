import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

@dataclass
class Position:
    symbol: str
    entry_price: float
    entry_date: pd.Timestamp
    quantity: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class TradeResult:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    return_pct: float

class Backtester:
    def __init__(self):
        self.positions: List[Position] = []
        self.trades: List[TradeResult] = []
        self.cash: float = 100000  # Initial capital
        self.equity: List[float] = []
        
    def reset(self, initial_capital: float = 100000):
        """Reset the backtester state"""
        self.positions = []
        self.trades = []
        self.cash = initial_capital
        self.equity = []
        
    def run_backtest(self, 
                    df: pd.DataFrame,
                    strategy_func: Callable,
                    stop_loss_pct: Optional[float] = None,
                    take_profit_pct: Optional[float] = None,
                    position_size_pct: float = 0.1,
                    commission_pct: float = 0.001) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            df: Historical price data
            strategy_func: Function that returns 1 (buy), -1 (sell), or 0 (hold)
            stop_loss_pct: Optional stop loss percentage
            take_profit_pct: Optional take profit percentage
            position_size_pct: Position size as percentage of capital
            commission_pct: Commission percentage per trade
        """
        self.reset()
        
        for i in range(len(df)):
            current_bar = df.iloc[i]
            
            # Update positions and check stops
            self._update_positions(current_bar, stop_loss_pct, take_profit_pct, commission_pct)
            
            # Get strategy signal
            signal = strategy_func(df.iloc[:i+1])
            
            # Execute trades based on signal
            if signal == 1 and not self.positions:  # Buy signal
                position_size = self.cash * position_size_pct
                quantity = int(position_size / current_bar['Close'])
                
                if quantity > 0:
                    self.positions.append(Position(
                        symbol=df.index.name or 'UNKNOWN',
                        entry_price=current_bar['Close'],
                        entry_date=current_bar.name,
                        quantity=quantity,
                        stop_loss=current_bar['Close'] * (1 - stop_loss_pct) if stop_loss_pct else None,
                        take_profit=current_bar['Close'] * (1 + take_profit_pct) if take_profit_pct else None
                    ))
                    self.cash -= quantity * current_bar['Close'] * (1 + commission_pct)
                    
            elif signal == -1 and self.positions:  # Sell signal
                self._close_positions(current_bar, commission_pct)
                
            # Track equity
            self.equity.append(self._calculate_equity(current_bar))
            
        return self._calculate_metrics(df)
    
    def _update_positions(self, 
                         current_bar: pd.Series,
                         stop_loss_pct: Optional[float],
                         take_profit_pct: Optional[float],
                         commission_pct: float):
        """Update positions and check for stops"""
        for position in self.positions[:]:
            # Check stop loss
            if position.stop_loss and current_bar['Low'] <= position.stop_loss:
                self._close_position(position, current_bar, commission_pct, position.stop_loss)
                continue
                
            # Check take profit
            if position.take_profit and current_bar['High'] >= position.take_profit:
                self._close_position(position, current_bar, commission_pct, position.take_profit)
                
    def _close_positions(self, current_bar: pd.Series, commission_pct: float):
        """Close all open positions"""
        for position in self.positions[:]:
            self._close_position(position, current_bar, commission_pct)
            
    def _close_position(self, 
                       position: Position,
                       current_bar: pd.Series,
                       commission_pct: float,
                       exit_price: Optional[float] = None):
        """Close a single position"""
        if exit_price is None:
            exit_price = current_bar['Close']
            
        pnl = (exit_price - position.entry_price) * position.quantity
        return_pct = (exit_price - position.entry_price) / position.entry_price
        
        self.trades.append(TradeResult(
            entry_date=position.entry_date,
            exit_date=current_bar.name,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            pnl=pnl,
            return_pct=return_pct
        ))
        
        self.cash += exit_price * position.quantity * (1 - commission_pct)
        self.positions.remove(position)
        
    def _calculate_equity(self, current_bar: pd.Series) -> float:
        """Calculate current equity"""
        position_value = sum(pos.quantity * current_bar['Close'] for pos in self.positions)
        return self.cash + position_value
        
    def _calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate backtest performance metrics"""
        if not self.trades:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0
            }
            
        equity_series = pd.Series(self.equity, index=df.index[:len(self.equity)])
        returns = equity_series.pct_change().dropna()
        
        # Calculate metrics
        total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0]
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
        
        # Calculate maximum drawdown
        peak = equity_series.expanding(min_periods=1).max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calculate win rate and profit factor
        winning_trades = [t for t in self.trades if t.pnl > 0]
        win_rate = len(winning_trades) / len(self.trades)
        
        gross_profits = sum(t.pnl for t in winning_trades)
        gross_losses = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        } 