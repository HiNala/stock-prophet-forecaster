"""
Stock Forecaster Application
Main entry point
"""

import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.app.stock_forecaster import StockForecaster

def main():
    """Initialize and run the application"""
    app = StockForecaster()
    app.mainloop()

if __name__ == "__main__":
    main() 