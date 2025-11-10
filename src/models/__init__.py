"""Model module."""

from .registry import register_model, get_model, list_models
from .architectures.unet import UNet
from .architectures.roinet import RoiNet
from .architectures.transunet import TransUNet
from .architectures.transroinet import TransRoiNet


def create_model(config: dict):
    """Create model from experiment configuration."""
    model_config = config['model']
    model_type = model_config['type']
    return get_model(model_type, model_config) 