from datetime import datetime, timedelta
from src.data_processor import DataProcessor
from src.data_manager import DataManager
from src.forecaster import Forecaster
from src.plotter import plot_forecast
from src.logger import logger

def main():
    """Main entry point for the stock price forecasting application."""
    try:
        # Configuration for technical indicators and data processing
        config = {
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
            },
            "data": {
                "cache_size": 1000,
                "cache_validity_minutes": 60,
                "max_workers": 4,
                "retry_delay_seconds": 5
            }
        }
        
        # Initialize components with configuration
        data_processor = DataProcessor(config=config)
        data_manager = DataManager(data_processor)
        forecaster = Forecaster(data_manager)
        
        # Fetch and process data
        symbol = "AAPL"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data
        
        forecast_data = forecaster.generate_forecast(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Plot results
        plot_forecast(forecast_data, symbol)
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 