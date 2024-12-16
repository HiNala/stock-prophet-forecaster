# Stock Price Forecaster

A Python application that combines stock price analysis with Facebook Prophet forecasting in a user-friendly GUI interface.

## Features

- Real-time stock data fetching using yfinance
- Interactive candlestick charts with technical indicators:
  - 20-day Simple Moving Average (SMA)
  - 50-day Exponential Moving Average (EMA)
  - Relative Strength Index (RSI)
- Advanced time series forecasting using Facebook Prophet
- Modern GUI interface using CustomTkinter
- Interactive plots using Plotly

## Installation

1. Make sure you have Python 3.8+ installed
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python stock_forecaster.py
```

2. Enter a stock ticker symbol (e.g., AAPL for Apple Inc.)
3. Click "Load Data" to view the current stock data and technical indicators
4. Click "Generate Forecast" to see the 90-day price prediction

## Technical Indicators

- **SMA (20-day)**: Shows the average closing price over the last 20 days
- **EMA (50-day)**: Gives more weight to recent prices, showing trend direction
- **RSI**: Measures momentum, with values above 70 indicating overbought conditions and below 30 indicating oversold conditions

## Notes

- The forecast model uses Facebook Prophet with daily, weekly, and yearly seasonality
- Charts will open in your default web browser for better interactivity
- Historical data is fetched for the last 2 years 

## Common Issues and Troubleshooting

### The 'ds' Column Issue

The 'ds' column error is a common issue when working with Prophet forecasts. Here's what you need to know:

1. **What is the 'ds' column?**
   - Prophet requires your date/timestamp column to be named 'ds'
   - The target variable must be named 'y'
   - These are non-negotiable requirements of the Prophet library

2. **Common Causes of 'ds' Errors:**
   - Model fitting failure causing empty forecast DataFrame
   - Date column not properly converted to datetime format
   - NaN values in the date column
   - Date column missing after data transformations
   - Index reset issues with pandas DataFrames

3. **How to Fix:**
   - Always verify your input data has a valid date column
   - Ensure date column is converted to datetime using `pd.to_datetime()`
   - Check for NaN values in both 'ds' and 'y' columns
   - Verify DataFrame operations maintain the 'ds' column
   - Use proper error handling around model fitting and forecast generation

4. **Prevention:**
   - Add validation checks before model fitting
   - Verify DataFrame columns after each transformation
   - Log DataFrame state at key points in the process
   - Handle edge cases (empty DataFrames, failed model fits)

5. **Debug Steps:**
   ```python
   # Verify date column
   print(df['ds'].dtype)  # Should be datetime64[ns]
   print(df['ds'].isna().sum())  # Should be 0
   
   # Check DataFrame integrity
   print(df.columns)  # Should include 'ds' and 'y'
   print(df.shape)  # Verify row count
   
   # Validate forecast output
   print(forecast.columns)  # Should include 'ds', 'yhat', etc.
   print(forecast.empty)  # Should be False
   ```

### Best Practices

1. **Data Preparation:**
   ```python
   # Always convert to datetime explicitly
   df['ds'] = pd.to_datetime(df['ds'])
   
   # Sort dates
   df = df.sort_values('ds')
   
   # Handle missing values
   df = df.dropna(subset=['ds', 'y'])
   ```

2. **Error Handling:**
   ```python
   # Validate forecast DataFrame
   if forecast is None or forecast.empty:
       logger.error("Forecast generation failed")
       return pd.DataFrame(), {}
   
   # Check required columns
   required_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
   if not all(col in forecast.columns for col in required_cols):
       logger.error("Missing required columns")
       return pd.DataFrame(), {}
   ```

3. **Debugging:**
   ```python
   # Add debug logging
   logger.debug(f"DataFrame shape: {df.shape}")
   logger.debug(f"Columns: {df.columns.tolist()}")
   logger.debug(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
   ```

Remember to check the logs when issues occur, as they often contain valuable information about where and why the process failed.