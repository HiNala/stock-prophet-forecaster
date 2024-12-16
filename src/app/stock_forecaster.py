"""
Main application class for the Stock Forecaster
"""

import customtkinter as ctk
from ..utils import get_logger, get_config, get_data_manager
from ..ui import apply_theme_to_widget, get_chart_style, get_status_color, ACTIVE_THEME, FONTS, PADDINGS
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os
from datetime import datetime, timedelta
import pandas as pd

logger = get_logger()
config = get_config()
data_manager = get_data_manager()

__all__ = ['StockForecaster']

class StockForecaster(ctk.CTk):
    def __init__(self):
        super().__init__()
        logger.debug("Initializing StockForecaster application")
        
        # Configure window
        self.title("Stock Price Forecaster")
        self.geometry(config.get('ui', 'window_size'))
        
        # Configure grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)
        
        # Create frames
        self.setup_frames()
        
        # Initialize data
        self.data = None
        self.forecast_data = None
        self.forecast_metrics = None
        
        # Create output directory
        self.output_dir = os.path.join('output', 'charts')
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.debug("Application initialization complete")
        
    def setup_frames(self):
        """Setup main application frames"""
        # Control panel (left side)
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.setup_control_panel()
        
        # Main panel (right side)
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.setup_main_panel()
        
    def setup_control_panel(self):
        """Setup the control panel with all user inputs"""
        # Stock Input Section
        stock_frame = ctk.CTkFrame(self.control_frame)
        stock_frame.pack(fill="x", padx=10, pady=5)
        
        header_label = ctk.CTkLabel(stock_frame, text="Stock Settings", font=FONTS['header'])
        header_label.pack(padx=5, pady=5)
        self._add_tooltip(header_label, "Configure stock data settings")
        
        # Ticker input with tooltip
        ticker_frame = ctk.CTkFrame(stock_frame)
        ticker_frame.pack(fill="x", padx=5, pady=2)
        
        self.ticker_entry = ctk.CTkEntry(ticker_frame, placeholder_text="Enter ticker (e.g., AAPL)")
        self.ticker_entry.pack(side="left", fill="x", expand=True, padx=(5,25))
        self._add_tooltip(self.ticker_entry, "Enter the stock symbol (e.g., AAPL for Apple Inc.)")
        
        # Time period selection with tooltip
        period_frame = ctk.CTkFrame(stock_frame)
        period_frame.pack(fill="x", padx=5, pady=2)
        
        period_label = ctk.CTkLabel(period_frame, text="Time Period:")
        period_label.pack(side="left", padx=5)
        
        self.period_var = ctk.StringVar(value="6mo")
        periods = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]
        period_menu = ctk.CTkOptionMenu(period_frame, values=periods, variable=self.period_var)
        period_menu.pack(side="left", padx=(5,25))
        self._add_tooltip(period_menu, "Choose how far back to load historical data")
        
        # Technical Indicators Section
        indicators_frame = ctk.CTkFrame(self.control_frame)
        indicators_frame.pack(fill="x", padx=10, pady=5)
        
        header_label = ctk.CTkLabel(indicators_frame, text="Technical Indicators", font=FONTS['header'])
        header_label.pack(padx=5, pady=5)
        self._add_tooltip(header_label, "Configure technical analysis indicators")
        
        # Moving Averages
        ma_frame = ctk.CTkFrame(indicators_frame)
        ma_frame.pack(fill="x", padx=5, pady=2)
        
        self.sma_var = ctk.BooleanVar(value=True)
        sma_check = ctk.CTkCheckBox(ma_frame, text="SMA", variable=self.sma_var)
        sma_check.pack(side="left", padx=5)
        self._add_tooltip(sma_check, "Simple Moving Average - Shows average price over a period")
        
        self.sma_period = ctk.StringVar(value="20")
        sma_entry = ctk.CTkEntry(ma_frame, width=50, textvariable=self.sma_period)
        sma_entry.pack(side="left", padx=(5,25))
        
        self.ema_var = ctk.BooleanVar(value=True)
        ema_check = ctk.CTkCheckBox(ma_frame, text="EMA", variable=self.ema_var)
        ema_check.pack(side="left", padx=5)
        self._add_tooltip(ema_check, "Exponential Moving Average - Weighted towards recent prices")
        
        self.ema_period = ctk.StringVar(value="50")
        ema_entry = ctk.CTkEntry(ma_frame, width=50, textvariable=self.ema_period)
        ema_entry.pack(side="left", padx=(5,25))
        
        # Other Indicators
        other_frame = ctk.CTkFrame(indicators_frame)
        other_frame.pack(fill="x", padx=5, pady=2)
        
        self.rsi_var = ctk.BooleanVar(value=True)
        rsi_check = ctk.CTkCheckBox(other_frame, text="RSI", variable=self.rsi_var)
        rsi_check.pack(side="left", padx=5)
        self._add_tooltip(rsi_check, "Relative Strength Index - Momentum indicator (30/70 boundaries)")
        
        self.macd_var = ctk.BooleanVar(value=True)
        macd_check = ctk.CTkCheckBox(other_frame, text="MACD", variable=self.macd_var)
        macd_check.pack(side="left", padx=5)
        self._add_tooltip(macd_check, "Moving Average Convergence/Divergence - Trend indicator")
        
        self.bbands_var = ctk.BooleanVar(value=True)
        bb_check = ctk.CTkCheckBox(other_frame, text="Bollinger", variable=self.bbands_var)
        bb_check.pack(side="left", padx=5)
        self._add_tooltip(bb_check, "Bollinger Bands - Volatility indicator (2 standard deviations)")
        
        self.stoch_var = ctk.BooleanVar(value=False)
        stoch_check = ctk.CTkCheckBox(other_frame, text="Stochastic", variable=self.stoch_var)
        stoch_check.pack(side="left", padx=5)
        self._add_tooltip(stoch_check, "Stochastic Oscillator - Momentum indicator comparing closing price to price range")
        
        # Forecast Settings Section
        forecast_frame = ctk.CTkFrame(self.control_frame)
        forecast_frame.pack(fill="x", padx=10, pady=5)
        
        header_label = ctk.CTkLabel(forecast_frame, text="Forecast Settings", font=FONTS['header'])
        header_label.pack(padx=5, pady=5)
        self._add_tooltip(header_label, "Configure price prediction settings")
        
        # Forecast period
        period_frame = ctk.CTkFrame(forecast_frame)
        period_frame.pack(fill="x", padx=5, pady=2)
        
        days_label = ctk.CTkLabel(period_frame, text="Days to Forecast:")
        days_label.pack(side="left", padx=5)
        
        self.forecast_days = ctk.StringVar(value="30")
        days_entry = ctk.CTkEntry(period_frame, width=50, textvariable=self.forecast_days)
        days_entry.pack(side="left", padx=(5,25))
        self._add_tooltip(days_entry, "Enter number of days (1-365)")
        
        # Action Buttons
        button_frame = ctk.CTkFrame(self.control_frame)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        self.load_button = ctk.CTkButton(
            button_frame,
            text="Load Data",
            command=self.load_data,
            fg_color=ACTIVE_THEME['primary_color'],
            hover_color=ACTIVE_THEME['secondary_color']
        )
        self.load_button.pack(fill="x", pady=2)
        self._add_tooltip(self.load_button, "Load and analyze historical stock data")
        
        self.forecast_button = ctk.CTkButton(
            button_frame,
            text="Generate Forecast",
            command=self.generate_forecast,
            fg_color=ACTIVE_THEME['primary_color'],
            hover_color=ACTIVE_THEME['secondary_color']
        )
        self.forecast_button.pack(fill="x", pady=2)
        self.forecast_button.configure(state="disabled")
        self._add_tooltip(self.forecast_button, "Generate price predictions using machine learning")
        
    def _add_tooltip(self, widget, text):
        """Add tooltip with info icon that appears on hover"""
        # Create tooltip frame that will be shown/hidden
        tooltip_frame = ctk.CTkFrame(
            self,
            fg_color=ACTIVE_THEME['tooltip_bg'],
            corner_radius=6,
            border_width=1
        )
        
        # Create tooltip content
        tooltip_label = ctk.CTkLabel(
            tooltip_frame,
            text=text,
            text_color=ACTIVE_THEME['tooltip_fg'],
            wraplength=300,
            justify="left",
            padx=10,
            pady=5,
            font=("Segoe UI", 11)
        )
        tooltip_label.pack(padx=5, pady=5)
        
        # Add info icon with transparent background
        info_label = ctk.CTkLabel(
            widget.master,
            text="ⓘ",
            font=("Segoe UI", 11),
            text_color=ACTIVE_THEME['primary_color'],
            fg_color="transparent",
            width=15,
            cursor="hand2"
        )
        
        # Position info icon next to the widget without disrupting layout
        if isinstance(widget, ctk.CTkEntry):
            info_label.place(in_=widget, relx=1.0, rely=0.5, anchor="e", x=-5)
        elif isinstance(widget, ctk.CTkOptionMenu):
            info_label.place(in_=widget, relx=1.0, rely=0.5, anchor="e", x=-5)
        elif isinstance(widget, ctk.CTkCheckBox):
            info_label.place(in_=widget, relx=1.0, rely=0.5, anchor="w", x=5)
        else:
            info_label.place(in_=widget, relx=1.0, rely=0.5, anchor="w", x=5)
        
        def show_tooltip(event):
            # Get widget's absolute position
            widget_x = event.widget.winfo_rootx()
            widget_y = event.widget.winfo_rooty()
            widget_height = event.widget.winfo_height()
            
            # Calculate initial position (prefer right side)
            x = widget_x + event.widget.winfo_width() + 10
            y = widget_y + (widget_height // 2)
            
            # Get screen dimensions
            screen_width = widget.winfo_screenwidth()
            screen_height = widget.winfo_screenheight()
            
            # Ensure tooltip stays within screen bounds
            tooltip_width = tooltip_frame.winfo_reqwidth()
            tooltip_height = tooltip_frame.winfo_reqheight()
            
            # Adjust horizontal position if needed
            if x + tooltip_width > screen_width:
                x = widget_x - tooltip_width - 10
            
            # Center tooltip vertically relative to widget
            y = y - (tooltip_height // 2)
            
            # Ensure tooltip stays within vertical screen bounds
            if y + tooltip_height > screen_height:
                y = screen_height - tooltip_height - 5
            elif y < 0:
                y = 5
            
            # Show tooltip
            tooltip_frame.place(x=x, y=y)
            tooltip_frame.lift()
        
        def hide_tooltip(event):
            tooltip_frame.place_forget()
        
        # Bind hover events
        widget.bind('<Enter>', show_tooltip)
        widget.bind('<Leave>', hide_tooltip)
        info_label.bind('<Enter>', show_tooltip)
        info_label.bind('<Leave>', hide_tooltip)
        
        return widget

    def setup_main_panel(self):
        """Setup the main panel with enhanced status and results"""
        # Status frame with improved styling
        status_frame = ctk.CTkFrame(self.main_frame)
        status_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="Ready",
            font=FONTS['header'],
            fg_color=ACTIVE_THEME['success_color'],
            corner_radius=6,
            padx=20,
            pady=10
        )
        self.status_label.pack(padx=10, pady=5)
        
        # Results frame with improved styling
        results_frame = ctk.CTkFrame(self.main_frame)
        results_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        results_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid_rowconfigure(1, weight=1)
        
        # Metrics frame with enhanced styling
        self.metrics_frame = ctk.CTkFrame(results_frame)
        self.metrics_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        
        # Debug frame with improved styling
        self.debug_frame = ctk.CTkFrame(results_frame)
        self.debug_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        
        self.debug_text = ctk.CTkTextbox(
            self.debug_frame,
            font=FONTS['monospace'],
            fg_color=ACTIVE_THEME['debug_bg'],
            text_color=ACTIVE_THEME['debug_fg']
        )
        self.debug_text.pack(fill="both", expand=True, padx=10, pady=10)
        
    def load_data(self):
        """Load and process stock data"""
        try:
            # Get ticker and validate
            ticker = self.ticker_entry.get().strip().upper()
            if not ticker:
                self.log_debug("Please enter a ticker symbol")
                return
                
            self.log_debug(f"Loading data for {ticker}...")
            self.status_label.configure(text=f"Loading {ticker} data...")
            
            # Map UI period selections to yfinance periods
            period_mapping = {
                "1mo": 30,
                "3mo": 90,
                "6mo": 180,
                "1y": 365,
                "2y": 730,
                "5y": 1825,
                "10y": 3650,
                "max": 7300  # 20 years as max
            }
            
            selected_period = self.period_var.get()
            days = period_mapping.get(selected_period, 180)  # Default to 6 months
            
            # Calculate start and end dates
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            self.log_debug(f"Selected period: {selected_period} ({days} days)")
            self.log_debug(f"Fetching data from {start_date.date()} to {end_date.date()}")
            
            # Fetch data with date range
            self.data = data_manager.fetch_data(
                ticker,
                start_date=start_date,
                end_date=end_date
            )
            
            if self.data is None or (isinstance(self.data, pd.DataFrame) and self.data.empty):
                self.log_debug("No data available for the selected period")
                self.status_label.configure(text="No data available")
                return
            
            # Verify we got the requested time period
            if isinstance(self.data, pd.DataFrame):
                actual_days = (self.data.index.max() - self.data.index.min()).days
                self.log_debug(f"Received {actual_days} days of data")
                if actual_days < days * 0.9:  # Allow for some missing days (weekends/holidays)
                    self.log_debug(f"Warning: Received less data than requested ({actual_days} vs {days} days)")
            
            # Calculate technical indicators
            selected_indicators = {
                'sma': self.sma_var.get(),
                'ema': self.ema_var.get(),
                'rsi': self.rsi_var.get(),
                'macd': self.macd_var.get(),
                'bollinger': self.bbands_var.get(),
                'stochastic': self.stoch_var.get()
            }
            
            self.data = data_manager.get_technical_indicators(self.data, selected_indicators)
            
            # Update status and enable forecast button
            self.status_label.configure(text=f"Loaded {ticker} data")
            self.forecast_button.configure(state="normal")
            
            # Create and display chart
            self.create_stock_chart()
            
            # Display metrics
            self.display_metrics()
            
        except Exception as e:
            self.log_debug(f"Error loading data: {str(e)}")
            self.status_label.configure(text="Error loading data")
            
    def create_stock_chart(self):
        """Create and display stock chart with indicators"""
        try:
            # Calculate number of subplots needed
            num_rows = 1  # Main price chart
            if self.rsi_var.get(): num_rows += 1
            if self.macd_var.get(): num_rows += 1
            if self.stoch_var.get(): num_rows += 1
            
            # Create subplot layout
            row_heights = [0.5 if i == 0 else 0.5/(num_rows-1) for i in range(num_rows)] if num_rows > 1 else [1]
            fig = make_subplots(rows=num_rows, cols=1, row_heights=row_heights, vertical_spacing=0.05)
            
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=self.data.index,
                    open=self.data['open_original'],
                    high=self.data['high'],
                    low=self.data['low'],
                    close=self.data['close_original'],
                    name='OHLC'
                ),
                row=1, col=1
            )
            
            # Add moving averages
            if self.sma_var.get():
                period = int(self.sma_period.get())
                sma = self.data['close_original'].rolling(window=period).mean()
                fig.add_trace(
                    go.Scatter(
                        x=self.data.index,
                        y=sma,
                        name=f'SMA {period}',
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )
            
            if self.ema_var.get():
                period = int(self.ema_period.get())
                ema = self.data['close_original'].ewm(span=period, adjust=False).mean()
                fig.add_trace(
                    go.Scatter(
                        x=self.data.index,
                        y=ema,
                        name=f'EMA {period}',
                        line=dict(color='orange')
                    ),
                    row=1, col=1
                )
            
            # Add Bollinger Bands
            if self.bbands_var.get():
                for band, color in [('upper', 'rgba(250,0,0,0.2)'), ('lower', 'rgba(250,0,0,0.2)'), ('middle', 'red')]:
                    fig.add_trace(
                        go.Scatter(
                            x=self.data.index,
                            y=self.data[f'bb_{band}'],
                            name=f'BB {band}',
                            line=dict(color=color)
                        ),
                        row=1, col=1
                    )
            
            # Add RSI
            current_row = 1
            if self.rsi_var.get():
                current_row += 1
                fig.add_trace(
                    go.Scatter(
                        x=self.data.index,
                        y=self.data['rsi'],
                        name='RSI',
                        line=dict(color='purple')
                    ),
                    row=current_row, col=1
                )
                fig.add_hline(y=70, line_color="red", line_dash="dash", row=current_row, col=1)
                fig.add_hline(y=30, line_color="green", line_dash="dash", row=current_row, col=1)
            
            # Add MACD
            if self.macd_var.get():
                current_row += 1
                fig.add_trace(
                    go.Scatter(
                        x=self.data.index,
                        y=self.data['macd'],
                        name='MACD',
                        line=dict(color='blue')
                    ),
                    row=current_row, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=self.data.index,
                        y=self.data['macd_signal'],
                        name='Signal',
                        line=dict(color='orange')
                    ),
                    row=current_row, col=1
                )
            
            # Add Stochastic
            if self.stoch_var.get():
                current_row += 1
                fig.add_trace(
                    go.Scatter(
                        x=self.data.index,
                        y=self.data['stoch_k'],
                        name='Stoch %K',
                        line=dict(color='blue')
                    ),
                    row=current_row, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=self.data.index,
                        y=self.data['stoch_d'],
                        name='Stoch %D',
                        line=dict(color='orange')
                    ),
                    row=current_row, col=1
                )
                fig.add_hline(y=80, line_color="red", line_dash="dash", row=current_row, col=1)
                fig.add_hline(y=20, line_color="green", line_dash="dash", row=current_row, col=1)
            
            # Update layout
            chart_style = get_chart_style()
            fig.update_layout(
                title=f'{self.ticker_entry.get().upper()} Stock Price Chart',
                xaxis_rangeslider_visible=False,
                height=800,
                **chart_style
            )
            
            # Save and display chart
            chart_path = os.path.join(self.output_dir, f"{self.ticker_entry.get().upper()}_chart.html")
            fig.write_html(chart_path)
            webbrowser.open(f'file://{os.path.abspath(chart_path)}')
            
        except Exception as e:
            self.log_debug(f"Error creating chart: {str(e)}")
            
    def generate_forecast(self):
        """Generate and display forecast"""
        try:
            if self.data is None:
                self.log_debug("Please load data first")
                return
                
            self.log_debug("Generating forecast...")
            self.status_label.configure(text="Generating forecast...")
            
            # Get forecast parameters
            days = int(self.forecast_days.get())
            
            # Generate forecast
            self.forecast_data, self.forecast_metrics = data_manager.generate_forecast(
                self.data,
                days=days,
                confidence=0.95
            )
            
            # Create forecast chart
            self.create_forecast_chart()
            
            # Display metrics
            self.display_metrics()
            
            self.status_label.configure(text="Forecast generated successfully")
            
        except Exception as e:
            self.log_debug(f"Error generating forecast: {str(e)}")
            self.status_label.configure(text="Error generating forecast")
            
    def create_forecast_chart(self):
        """Create and display forecast chart with improved visualization"""
        try:
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['close_original'],
                    name='Historical',
                    line=dict(color='blue', width=2)
                )
            )
            
            # Forecast with improved visibility
            fig.add_trace(
                go.Scatter(
                    x=self.forecast_data['ds'],
                    y=self.forecast_data['yhat'],
                    name='Forecast',
                    line=dict(color='red', width=2, dash='dot')  # Made dotted for better distinction
                )
            )
            
            # Confidence interval with improved visibility
            fig.add_trace(
                go.Scatter(
                    name='Upper Bound',
                    x=self.forecast_data['ds'],
                    y=self.forecast_data['yhat_upper'],
                    mode='lines',
                    line=dict(color='rgba(255, 0, 0, 0.1)'),
                    showlegend=True
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    name='Lower Bound',
                    x=self.forecast_data['ds'],
                    y=self.forecast_data['yhat_lower'],
                    mode='lines',
                    line=dict(color='rgba(255, 0, 0, 0.1)'),
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    showlegend=True
                )
            )
            
            # Calculate y-axis range to ensure confidence intervals are visible
            y_min = min(
                self.data['close_original'].min(),
                self.forecast_data['yhat_lower'].min()
            ) * 0.95  # Add 5% padding
            
            y_max = max(
                self.data['close_original'].max(),
                self.forecast_data['yhat_upper'].max()
            ) * 1.05  # Add 5% padding
            
            # Update layout with improved styling
            chart_style = get_chart_style()
            fig.update_layout(
                title=f'{self.ticker_entry.get().upper()} Price Forecast',
                xaxis_title='Date',
                yaxis_title='Price',
                height=800,
                hovermode='x unified',
                yaxis_range=[y_min, y_max],  # Set fixed y-axis range
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(255, 255, 255, 0.8)'
                ),
                **chart_style
            )
            
            # Update axes with improved grid
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                showspikes=True,  # Add spike lines
                spikethickness=1,
                spikecolor="gray",
                spikemode="across"
            )
            
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                showspikes=True,  # Add spike lines
                spikethickness=1,
                spikecolor="gray",
                spikemode="across"
            )
            
            # Save and display chart
            forecast_path = os.path.join(self.output_dir, f"{self.ticker_entry.get().upper()}_forecast.html")
            fig.write_html(forecast_path)
            webbrowser.open(f'file://{os.path.abspath(forecast_path)}')
            
        except Exception as e:
            self.log_debug(f"Error creating forecast chart: {str(e)}")
            
    def display_metrics(self):
        """Display current metrics"""
        try:
            # Clear previous metrics
            for widget in self.metrics_frame.winfo_children():
                widget.destroy()
            
            # Get metrics
            metrics = data_manager.get_summary_stats(self.data)
            
            # Display data quality metrics
            quality = metrics.get('data_quality', {})
            if quality:
                ctk.CTkLabel(
                    self.metrics_frame,
                    text=f"Data Points: {quality.get('total_rows', 'N/A')}",
                    font=FONTS['default']
                ).pack(side="left", padx=5)
            
            # Display trend metrics
            trend = metrics.get('trend_analysis', {})
            if trend:
                trend_color = get_status_color('success' if trend.get('direction') == 'Upward' else 'error')
                ctk.CTkLabel(
                    self.metrics_frame,
                    text=f"Trend: {trend.get('direction', 'N/A')} ({trend.get('strength', 0):.2f})",
                    font=FONTS['default'],
                    text_color=trend_color
                ).pack(side="left", padx=5)
            
            # Display forecast metrics if available
            if self.forecast_metrics:
                ctk.CTkLabel(
                    self.metrics_frame,
                    text=f"Forecast: {self.forecast_metrics.get('pred_price', 'N/A')} "
                         f"({self.forecast_metrics.get('change', 'N/A')})",
                    font=FONTS['default']
                ).pack(side="left", padx=5)
                
                ctk.CTkLabel(
                    self.metrics_frame,
                    text=f"Accuracy: {self.forecast_metrics.get('accuracy', 'N/A')}",
                    font=FONTS['default']
                ).pack(side="left", padx=5)
            
        except Exception as e:
            self.log_debug(f"Error displaying metrics: {str(e)}")
            
    def log_debug(self, message):
        """Log debug message to console and UI"""
        logger.debug(message)
        self.debug_text.insert("end", f"{datetime.now().strftime('%H:%M:%S')}: {message}\n")
        self.debug_text.see("end") 