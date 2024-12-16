"""
Logging configuration for the application
"""

import logging
import os
from datetime import datetime

class Logger:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance
    
    def _initialize_logger(self):
        """Initialize the logger with console and file handlers"""
        self.logger = logging.getLogger('StockForecaster')
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatters
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(console_formatter)
        
        # File handler
        log_dir = os.path.join('output', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'stock_forecaster_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    @classmethod
    def get_logger(cls):
        """Get the logger instance"""
        if cls._instance is None:
            cls()
        return cls._instance.logger

# Create a convenience function
def get_logger():
    """Get the application logger"""
    return Logger.get_logger() 