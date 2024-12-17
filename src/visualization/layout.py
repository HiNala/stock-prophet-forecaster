from dash import html, dcc
import dash_bootstrap_components as dbc

def create_layout():
    """Create the main application layout"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Stock Market Analysis & Forecasting", className="text-center mb-4"),
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Input(
                                    id="stock-input",
                                    type="text",
                                    placeholder="Enter stock symbol (e.g., AAPL)",
                                    className="mb-2"
                                ),
                            ], width=8),
                            dbc.Col([
                                dbc.Button(
                                    "Load Data",
                                    id="load-button",
                                    color="primary",
                                    className="w-100"
                                ),
                            ], width=4),
                        ]),
                    ])
                ], className="mb-4"),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Price Chart"),
                    dbc.CardBody([
                        dcc.Graph(id="price-chart"),
                        dbc.Row([
                            dbc.Col([
                                dcc.Dropdown(
                                    id="timeframe-dropdown",
                                    options=[
                                        {"label": "1 Month", "value": "1mo"},
                                        {"label": "3 Months", "value": "3mo"},
                                        {"label": "6 Months", "value": "6mo"},
                                        {"label": "1 Year", "value": "1y"},
                                        {"label": "2 Years", "value": "2y"},
                                        {"label": "5 Years", "value": "5y"},
                                    ],
                                    value="1y",
                                    className="mb-2"
                                ),
                            ], width=6),
                            dbc.Col([
                                dcc.Dropdown(
                                    id="indicator-dropdown",
                                    options=[
                                        {"label": "SMA", "value": "sma"},
                                        {"label": "EMA", "value": "ema"},
                                        {"label": "Bollinger Bands", "value": "bb"},
                                        {"label": "RSI", "value": "rsi"},
                                        {"label": "MACD", "value": "macd"},
                                    ],
                                    value=["sma"],
                                    multi=True,
                                    className="mb-2"
                                ),
                            ], width=6),
                        ]),
                    ]),
                ], className="mb-4"),
            ], width=12),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Prophet Forecast"),
                    dbc.CardBody([
                        dcc.Graph(id="forecast-chart"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Input(
                                    id="forecast-days",
                                    type="number",
                                    placeholder="Forecast days",
                                    value=30,
                                    min=1,
                                    max=365,
                                    className="mb-2"
                                ),
                            ], width=6),
                            dbc.Col([
                                dbc.Button(
                                    "Generate Forecast",
                                    id="forecast-button",
                                    color="success",
                                    className="w-100"
                                ),
                            ], width=6),
                        ]),
                    ]),
                ], className="mb-4"),
            ], width=12),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Backtesting"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dcc.Dropdown(
                                    id="strategy-dropdown",
                                    options=[
                                        {"label": "SMA Crossover", "value": "sma_cross"},
                                        {"label": "RSI Strategy", "value": "rsi"},
                                        {"label": "MACD Strategy", "value": "macd"},
                                    ],
                                    value="sma_cross",
                                    className="mb-2"
                                ),
                            ], width=6),
                            dbc.Col([
                                dbc.Button(
                                    "Run Backtest",
                                    id="backtest-button",
                                    color="warning",
                                    className="w-100"
                                ),
                            ], width=6),
                        ]),
                        html.Div(id="backtest-results", className="mt-3"),
                    ]),
                ], className="mb-4"),
            ], width=12),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Company Information"),
                    dbc.CardBody(id="company-info"),
                ]),
            ], width=12),
        ]),
        
        dcc.Store(id="stock-data"),
        dcc.Store(id="forecast-data"),
        dcc.Store(id="backtest-data"),
        
    ], fluid=True, className="mt-4") 