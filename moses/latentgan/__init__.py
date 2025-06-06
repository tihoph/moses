from .config import get_parser as latentGAN_parser
from .model import LatentGAN
from .trainer import LatentGANTrainer

__all__ = ["LatentGAN", "LatentGANTrainer", "latentGAN_parser"]
