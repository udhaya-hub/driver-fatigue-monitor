"""
Configuration loader utility.
"""

import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    path = Path(config_path)

    if not path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return get_default_config()

    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse config: {e}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Return default configuration."""
    return {
        "model": {
            "cnn": {
                "backbone": "efficientnet_b0",
                "pretrained": True,
                "freeze_backbone": False,
                "output_dim": 256,
                "dropout": 0.3,
            },
            "temporal": {
                "type": "lstm",
                "input_dim": 512,
                "hidden_dim": 128,
                "num_layers": 2,
                "bidirectional": True,
                "dropout": 0.2,
            },
            "transformer": {
                "d_model": 256,
                "nhead": 8,
                "num_encoder_layers": 3,
                "num_decoder_layers": 3,
                "dim_feedforward": 512,
                "dropout": 0.1,
                "activation": "gelu",
            },
            "heads": {
                "fatigue_classification": {
                    "num_classes": 3,
                    "hidden_dims": [128, 64],
                    "dropout": 0.3,
                },
                "workload_regression": {
                    "output_dim": 1,
                    "hidden_dims": [128, 64],
                    "dropout": 0.3,
                },
                "blink_detection": {
                    "num_classes": 2,
                    "hidden_dims": [64, 32],
                    "dropout": 0.2,
                },
            },
        },
        "vision": {
            "mediapipe": {
                "max_num_faces": 1,
                "refine_landmarks": True,
                "min_detection_confidence": 0.7,
                "min_tracking_confidence": 0.7,
            },
            "perclos": {
                "ear_threshold": 0.25,
                "consecutive_frames": 3,
                "window_size": 60,
            },
            "preprocessing": {
                "target_size": [224, 224],
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225],
            },
        },
        "physiological": {
            "sampling_rate": 64,
            "window_size": 30,
            "overlap": 0.5,
            "features": [
                "hrv_time_domain",
                "hrv_frequency_domain",
                "eda_tonic",
                "eda_phasic",
                "respiration_rate",
                "temperature",
            ],
        },
        "inference": {
            "device": "auto",
            "batch_size": 1,
            "sequence_length": 30,
            "confidence_threshold": 0.7,
            "smoothing_window": 5,
            "inference_interval": 5,
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 1,
            "timeout": 30,
            "cors_origins": ["http://localhost:8501", "http://localhost:3000"],
        },
        "frontend": {
            "type": "streamlit",
            "port": 8501,
            "theme": "dark",
        },
        "logging": {
            "level": "INFO",
            "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            "file": "logs/app.log",
            "rotation": "10 MB",
            "retention": "7 days",
        },
    }


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge two configuration dictionaries.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result