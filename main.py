import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from src.visualization.layout import create_layout
from src.data.market_data import MarketDataManager
from src.models.prophet_model import ProphetModel
from src.backtesting.backtester import Backtester

def main():
    # Initialize components
    market_data = MarketDataManager()
    prophet_model = ProphetModel()
    backtester = Backtester()

    # Initialize Dash app
    app = dash.Dash(__name__, 
                   external_stylesheets=[dbc.themes.DARKLY],
                   title="Stock Market Analysis & Forecasting")
    
    # Create app layout
    app.layout = create_layout()
    
    # Start the application
    app.run_server(debug=True)

if __name__ == "__main__":
    main() 