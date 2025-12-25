import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.vehicle import VehicleConfig


class ConfigLoader:
    """Load and manage configuration files."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader.
        
        Args:
            config_path: Path to config file. Uses default if None.
        """
        if config_path is None:
            # Default config path
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "default_config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            print(f"Warning: Config file not found at {self.config_path}")
            print("Using default configuration")
            return self._get_default_config()
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "simulation": {
                "fps": 60,
                "screen_width": 800,
                "screen_height": 600,
                "dt": 0.1
            },
            "vehicle": {
                "length": 4.0,
                "width": 2.0,
                "wheelbase": 2.5,
                "max_speed": 10.0,
                "max_acceleration": 3.0,
                "max_deceleration": -5.0,
                "max_steering_angle": 0.6
            },
            "map": {
                "width": 100,
                "height": 100,
                "grid_size": 0.5,
                "obstacle_inflation": 0.3
            },
            "planner": {
                "algorithm": "astar",
                "goal_threshold": 1.0,
                "max_iterations": 10000
            },
            "controller": {
                "type": "pid",
                "kp": 1.0,
                "ki": 0.1,
                "kd": 0.5,
                "lookahead_distance": 5.0
            },
            "training": {
                "algorithm": "ppo",
                "total_timesteps": 100000,
                "learning_rate": 0.0003,
                "batch_size": 64,
                "gamma": 0.99
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Config key in dot notation (e.g., 'vehicle.max_speed')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_vehicle_config(self) -> VehicleConfig:
        """Create VehicleConfig from loaded configuration."""
        vehicle_cfg = self.config.get("vehicle", {})
        
        return VehicleConfig(
            length=vehicle_cfg.get("length", 4.0),
            width=vehicle_cfg.get("width", 2.0),
            wheelbase=vehicle_cfg.get("wheelbase", 2.5),
            max_velocity=vehicle_cfg.get("max_velocity", 10.0),
            max_acceleration=vehicle_cfg.get("max_acceleration", 3.0),
            max_deceleration=vehicle_cfg.get("max_deceleration", -5.0),
            max_steering_angle=vehicle_cfg.get("max_steering_angle", 0.6)
        )
    
    def get_simulation_params(self) -> Dict[str, Any]:
        """Get simulation parameters."""
        return self.config.get("simulation", {})
    
    def get_map_params(self) -> Dict[str, Any]:
        """Get map parameters."""
        return self.config.get("map", {})
    
    def get_planner_params(self) -> Dict[str, Any]:
        """Get planner parameters."""
        return self.config.get("planner", {})
    
    def get_controller_params(self) -> Dict[str, Any]:
        """Get controller parameters."""
        return self.config.get("controller", {})
    
    def save_config(self, output_path: Optional[str] = None):
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save config. Uses original path if None.
        """
        if output_path is None:
            output_path = self.config_path
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def update(self, key: str, value: Any):
        """
        Update configuration value.
        
        Args:
            key: Config key in dot notation
            value: New value
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def __repr__(self) -> str:
        return f"ConfigLoader(path={self.config_path})"