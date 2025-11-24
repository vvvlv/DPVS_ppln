"""Model module."""

from .registry import register_model, get_model, list_models
from .architectures.unet import UNet
from .architectures.roinet import RoiNet
from .architectures.utrans import UTrans
from .architectures.transroinet import TransRoiNet
from .architectures.tiny_swin import TinySwinUNet


def create_model(config: dict):
    """Create model from experiment configuration."""
    model_config = config['model']
    model_type = model_config['type']
    return get_model(model_type, model_config) 