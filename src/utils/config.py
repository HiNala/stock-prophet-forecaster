"""
Configuration management for the application
"""

import json
import os
from .logger import get_logger

logger = get_logger()

class Config:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_config()
        return cls._instance
    
    def _initialize_config(self):
        """Initialize configuration with defaults"""
        self.config_dir = os.path.join('output', 'settings')
        os.makedirs(self.config_dir, exist_ok=True)
        self.config_file = os.path.join(self.config_dir, 'config.json')
        
        # Default configuration
        self.defaults = {
            'ui': {
                'theme': 'dark',
                'window_size': '1200x800'
            },
            'chart': {
                'default_period': '2y',
                'chart_height': 800,
                'template': 'plotly_dark'
            },
            'technical': {
                'sma_period': 20,
                'ema_period': 50,
                'default_indicators': ['SMA', 'EMA', 'RSI']
            },
            'forecast': {
                'days': 90,
                'changepoint_scale': 0.05,
                'seasonality': {
                    'yearly': True,
                    'weekly': True,
                    'daily': True
                }
            },
            'data': {
                'cache_enabled': True,
                'cache_duration': 86400  # 24 hours in seconds
            }
        }
        
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file or create with defaults"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                logger.debug("Configuration loaded from file")
                return self._merge_with_defaults(config)
            else:
                logger.debug("No configuration file found, using defaults")
                return self.defaults.copy()
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return self.defaults.copy()
    
    def _merge_with_defaults(self, config):
        """Merge loaded configuration with defaults to ensure all keys exist"""
        merged = self.defaults.copy()
        
        def merge_dict(source, destination):
            for key, value in source.items():
                if key in destination:
                    if isinstance(value, dict):
                        merge_dict(value, destination[key])
                    else:
                        destination[key] = value
            return destination
        
        return merge_dict(config, merged)
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.debug("Configuration saved to file")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
    
    def get(self, section, key=None):
        """Get configuration value"""
        try:
            if key is None:
                return self.config[section]
            return self.config[section][key]
        except KeyError:
            logger.error(f"Configuration key not found: {section}.{key}")
            return None
    
    def set(self, section, key, value):
        """Set configuration value"""
        try:
            if section not in self.config:
                self.config[section] = {}
            self.config[section][key] = value
            self.save_config()
        except Exception as e:
            logger.error(f"Error setting configuration: {str(e)}")
    
    @classmethod
    def get_instance(cls):
        """Get the configuration instance"""
        if cls._instance is None:
            cls()
        return cls._instance

# Create a convenience function
def get_config():
    """Get the application configuration"""
    return Config.get_instance() 