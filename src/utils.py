import os
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

PROJECT_PATH = Path(Path(__file__).absolute()).parent.parent
