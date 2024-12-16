# Stock Forecaster Project Structure

```
stock_forecaster/
│
├── main.py                     # Entry point
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── app/                    # Application core
│   │   ├── __init__.py
│   │   └── stock_forecaster.py # Main application class
│   │
│   ├── ui/                     # UI components
│   │   ├── __init__.py
│   │   ├── control_panel.py    # Left panel controls
│   │   ├── main_panel.py       # Right panel (debug/status)
│   │   └── styles.py          # UI styles and constants
│   │
│   ├── analysis/              # Analysis modules
│   │   ├── __init__.py
│   │   ├── technical.py       # Technical indicators
│   │   ├── forecast.py        # Prophet forecasting
│   │   └── preprocessing.py   # Data preprocessing
│   │
│   ├── visualization/         # Visualization modules
│   │   ├── __init__.py
│   │   ├── charts.py         # Chart creation
│   │   └── plotting.py       # Plotting utilities
│   │
│   └── utils/                 # Utility modules
│       ├── __init__.py
│       ├── config.py         # Configuration management
│       ├── logger.py         # Logging setup
│       └── data_manager.py   # Data handling
│
├── output/                    # Output directory
│   ├── charts/               # Generated charts
│   ├── data/                 # Cached data
│   └── settings/             # User settings
│
└── tests/                    # Test directory
    ├── __init__.py
    ├── test_technical.py
    ├── test_forecast.py
    └── test_data_manager.py
```

## Module Responsibilities

### Main Components:
1. `main.py`: Application entry point and initialization
2. `src/app/stock_forecaster.py`: Main application class, orchestrates components

### UI Components:
1. `ui/control_panel.py`: All control panel widgets and logic
2. `ui/main_panel.py`: Debug and status panel
3. `ui/styles.py`: Centralized UI styling

### Analysis:
1. `analysis/technical.py`: Technical indicator calculations
2. `analysis/forecast.py`: Prophet model and forecasting
3. `analysis/preprocessing.py`: Data cleaning and preparation

### Visualization:
1. `visualization/charts.py`: Chart generation
2. `visualization/plotting.py`: Plotting utilities

### Utilities:
1. `utils/config.py`: Configuration management
2. `utils/logger.py`: Logging setup
3. `utils/data_manager.py`: Data handling and caching

### Benefits of this Structure:
1. Modular development
2. Easy testing
3. Clear separation of concerns
4. Maintainable codebase
5. Scalable architecture 