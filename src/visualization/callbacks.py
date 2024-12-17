from dash import Input, Output, State, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
from typing import List, Dict

from ..data.market_data import MarketDataManager
from ..models.prophet_model import ProphetModel
from ..backtesting.backtester import Backtester

# Initialize components
market_data = MarketDataManager()
prophet_model = ProphetModel()
backtester = Backtester()

@callback(
    [Output("stock-data", "data"),
     Output("price-chart", "figure"),
     Output("company-info", "children")],
    [Input("load-button", "n_clicks")],
    [State("stock-input", "value"),
     State("timeframe-dropdown", "value"),
     State("indicator-dropdown", "value")],
    prevent_initial_call=True
)
def load_stock_data(n_clicks, symbol, timeframe, indicators):
    """Load and display stock data"""
    if not symbol:
        return None, go.Figure(), "Please enter a stock symbol"
        
    # Fetch stock data
    df = market_data.get_stock_data(symbol, period=timeframe)
    if df is None:
        return None, go.Figure(), "Error loading stock data"
        
    # Create figure
    fig = create_stock_chart(df, indicators)
    
    # Get company info
    info = market_data.get_company_info(symbol)
    info_div = create_company_info_div(info)
    
    return df.to_json(date_format='iso'), fig, info_div

@callback(
    [Output("forecast-chart", "figure"),
     Output("forecast-data", "data")],
    [Input("forecast-button", "n_clicks")],
    [State("stock-data", "data"),
     State("forecast-days", "value")],
    prevent_initial_call=True
)
def generate_forecast(n_clicks, stock_data_json, forecast_days):
    """Generate and display forecast"""
    if not stock_data_json:
        return go.Figure(), None
        
    # Load stock data
    df = pd.read_json(stock_data_json)
    
    # Train model and generate forecast
    prophet_model.train(df)
    forecast_df, components = prophet_model.forecast(periods=forecast_days)
    
    # Create forecast figure
    fig = create_forecast_chart(df, forecast_df, components)
    
    return fig, forecast_df.to_json(date_format='iso')

@callback(
    [Output("backtest-results", "children"),
     Output("backtest-data", "data")],
    [Input("backtest-button", "n_clicks")],
    [State("stock-data", "data"),
     State("strategy-dropdown", "value")],
    prevent_initial_call=True
)
def run_backtest(n_clicks, stock_data_json, strategy):
    """Run and display backtest results"""
    if not stock_data_json:
        return "Please load stock data first", None
        
    # Load stock data
    df = pd.read_json(stock_data_json)
    
    # Get strategy function
    strategy_func = get_strategy_function(strategy)
    
    # Run backtest
    results = backtester.run_backtest(
        df=df,
        strategy_func=strategy_func,
        stop_loss_pct=0.02,
        take_profit_pct=0.05
    )
    
    # Create results div
    results_div = create_backtest_results_div(results)
    
    return results_div, json.dumps(results)

def create_stock_chart(df: pd.DataFrame, indicators: List[str]) -> go.Figure:
    """Create stock price chart with indicators"""
    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                       shared_xaxes=True, vertical_spacing=0.05)
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="OHLC"
        ),
        row=1, col=1
    )
    
    # Add indicators
    if 'sma' in indicators:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_20'], name="SMA 20",
                      line=dict(color='blue')), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_50'], name="SMA 50",
                      line=dict(color='orange')), row=1, col=1)
                      
    if 'bb' in indicators:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Upper'], name="BB Upper",
                      line=dict(color='gray', dash='dash')), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Lower'], name="BB Lower",
                      line=dict(color='gray', dash='dash')), row=1, col=1)
                      
    # Add volume bars
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name="Volume"),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title="Stock Price Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis2_title="Volume",
        showlegend=True,
        height=800
    )
    
    return fig

def create_forecast_chart(df: pd.DataFrame, forecast_df: pd.DataFrame,
                         components: Dict) -> go.Figure:
    """Create forecast chart with components"""
    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                       shared_xaxes=True, vertical_spacing=0.05)
    
    # Historical data
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Close'], name="Historical",
                  line=dict(color='blue')), row=1, col=1)
    
    # Forecast
    fig.add_trace(
        go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'],
                  name="Forecast", line=dict(color='red')), row=1, col=1)
    
    # Uncertainty intervals
    fig.add_trace(
        go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'],
                  fill=None, mode='lines', line_color='rgba(255,0,0,0.2)',
                  name='Upper Bound'), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'],
                  fill='tonexty', mode='lines', line_color='rgba(255,0,0,0.2)',
                  name='Lower Bound'), row=1, col=1)
    
    # Components
    fig.add_trace(
        go.Scatter(x=forecast_df['ds'], y=components['trend'],
                  name="Trend", line=dict(color='green')), row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title="Prophet Forecast",
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis2_title="Components",
        showlegend=True,
        height=800
    )
    
    return fig

def create_company_info_div(info: Dict) -> str:
    """Create company information HTML"""
    return f"""
        <div>
            <h4>{info.get('name', 'N/A')}</h4>
            <p><strong>Sector:</strong> {info.get('sector', 'N/A')}</p>
            <p><strong>Industry:</strong> {info.get('industry', 'N/A')}</p>
            <p><strong>Market Cap:</strong> ${info.get('market_cap', 0):,.2f}</p>
            <p><strong>P/E Ratio:</strong> {info.get('pe_ratio', 'N/A')}</p>
            <p>{info.get('description', '')}</p>
        </div>
    """

def create_backtest_results_div(results: Dict) -> str:
    """Create backtest results HTML"""
    return f"""
        <div>
            <h4>Backtest Results</h4>
            <p><strong>Total Return:</strong> {results['total_return']*100:.2f}%</p>
            <p><strong>Sharpe Ratio:</strong> {results['sharpe_ratio']:.2f}</p>
            <p><strong>Max Drawdown:</strong> {results['max_drawdown']*100:.2f}%</p>
            <p><strong>Win Rate:</strong> {results['win_rate']*100:.2f}%</p>
            <p><strong>Profit Factor:</strong> {results['profit_factor']:.2f}</p>
        </div>
    """

def get_strategy_function(strategy: str) -> callable:
    """Return strategy function based on selection"""
    def sma_crossover_strategy(df: pd.DataFrame) -> int:
        if len(df) < 50:
            return 0
        if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] and \
           df['SMA_20'].iloc[-2] <= df['SMA_50'].iloc[-2]:
            return 1
        elif df['SMA_20'].iloc[-1] < df['SMA_50'].iloc[-1] and \
             df['SMA_20'].iloc[-2] >= df['SMA_50'].iloc[-2]:
            return -1
        return 0
        
    def rsi_strategy(df: pd.DataFrame) -> int:
        if len(df) < 14:
            return 0
        if df['RSI'].iloc[-1] < 30:
            return 1
        elif df['RSI'].iloc[-1] > 70:
            return -1
        return 0
        
    def macd_strategy(df: pd.DataFrame) -> int:
        if len(df) < 26:
            return 0
        if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] and \
           df['MACD'].iloc[-2] <= df['MACD_Signal'].iloc[-2]:
            return 1
        elif df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1] and \
             df['MACD'].iloc[-2] >= df['MACD_Signal'].iloc[-2]:
            return -1
        return 0
        
    strategies = {
        'sma_cross': sma_crossover_strategy,
        'rsi': rsi_strategy,
        'macd': macd_strategy
    }
    
    return strategies.get(strategy, sma_crossover_strategy) 