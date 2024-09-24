from typing import Any, Dict, List, Optional, Tuple, Type, Union, Literal, Callable

import torch
from dataclasses import dataclass, asdict


@dataclass
class NoneConfig:

    _target_: str = "torch.nn.Identity"
