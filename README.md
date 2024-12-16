# Stock Prophet Forecaster

Advanced stock market forecasting tool combining Facebook Prophet with technical analysis. Features real-time market data processing, dynamic volatility detection, and parallel stock analysis. Built with Python, offering accurate price predictions with interactive visualizations.

## Features
- Advanced time series forecasting using Prophet
- Real-time technical indicator analysis
- Dynamic volatility regime detection
- Parallel processing for multiple stock analysis
- Interactive visualization of forecasts and trends
- Comprehensive error handling and validation
- Modular and extensible architecture

## Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git

## Quick Start Guide

1. Clone the repository:
```bash
git clone https://github.com/HiNala/stock-prophet-forecaster.git
cd stock-prophet-forecaster
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python main.py
```

## Usage Guide

1. After launching, enter a stock symbol (e.g., AAPL, MSFT, GOOGL)
2. Select your desired time period for analysis
3. Click "Load Data" to fetch and process stock data
4. View the interactive charts showing:
   - Historical price data with technical indicators
   - Forecast predictions with confidence intervals
   - Volatility regime analysis

## Project Structure
```
stock-prophet-forecaster/
├── src/                    # Source code
│   ├── analysis/          # Core analysis modules
│   ├── app/               # Application logic
│   ├── ui/                # User interface
│   ├── utils/             # Utilities
│   └── visualization/     # Visualization tools
├── main.py                # Application entry point
└── requirements.txt       # Project dependencies
```

## Troubleshooting

1. If you encounter package installation issues:
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

2. If Prophet installation fails, try:
```bash
# Windows
conda install -c conda-forge prophet

# Linux/Mac
pip install prophet --no-cache-dir
```

3. For visualization issues:
- Ensure you have a modern web browser installed
- Charts will open in your default browser

## License
MIT License
