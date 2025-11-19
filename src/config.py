"""
Configuration loader and utilities
"""
import yaml
import os
from typing import Dict, Any
import logging

class Config:
    """Configuration class to load and manage training parameters"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from YAML file
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logging()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        required_sections = ['dataset', 'models', 'training', 'output']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section '{section}' in config file")
                
        return config
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to the configuration value (e.g., 'dataset.batch_size')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to the configuration value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
            
        config[keys[-1]] = value
    
    def save(self, output_path: str = None):
        """
        Save current configuration to YAML file
        
        Args:
            output_path: Path to save the configuration. If None, overwrites original file
        """
        output_path = output_path or self.config_path
        with open(output_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)
    
    def create_directories(self):
        """Create necessary output directories"""
        directories = [
            self.get('output.models_dir'),
            self.get('output.logs_dir'),
        ]
        
        for directory in directories:
            if directory:
                os.makedirs(directory, exist_ok=True)
    
    def __getitem__(self, key):
        """Allow dictionary-style access"""
        return self.config[key]
    
    def __contains__(self, key):
        """Check if key exists in config"""
        return key in self.config