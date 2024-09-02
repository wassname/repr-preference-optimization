# from lightning.fabric.plugins.precision.precision import Precision
from lightning.pytorch.plugins.precision.precision import Precision
from lightning.fabric.plugins.precision.utils import (
    _ClassReplacementContextManager,
    _convert_fp_tensor,
    _DtypeContextManager,
)
from typing_extensions import Self, override
from lightning.fabric.utilities.types import _DEVICE
from torch import Tensor
from types import ModuleType
from contextlib import ExitStack
from lightning_utilities import apply_to_collection
from typing import Any, Callable, ContextManager, Literal, Optional, OrderedDict, Set, Tuple, Type, cast
import torch
from lightning.fabric.plugins.precision.bitsandbytes import _import_bitsandbytes


class ExistingBitsandbytesPrecision(Precision):
    """Plugin for already quantizing weights from `bitsandbytes <https://github.com/TimDettmers/bitsandbytes>`__.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    .. note::
        The optimizer is NOT automatically replaced with ``bitsandbytes.optim.Adam8bit`` or equivalent 8-bit optimizers.

    Args:
        dtype: The compute dtype to use.
    """

    # Note: you'll notice that the `precision` str class attribute is not defined. This is on purpose because there are
    # many configuration options so `precision="bitsandbytes"` would be ambiguous about which one to use. Additionally,
    # it would create backwards compatibility challenges if better modes or dtypes are added in the future

    # TODO: we could implement optimizer replacement with
    # - Fabric: Add `Precision.convert_optimizer` from `Strategy.setup_optimizer`
    # - Trainer: Use `Precision.connect`

    precision: str = "existing-bitsandbytes"

    def __init__(
        self,
        mode: Literal["nf4", "nf4-dq", "fp4", "fp4-dq", "int8", "int8-training"] = None,
        dtype: Optional[torch.dtype] = None,
        default_dtype: Optional[torch.dtype] = None,
    ) -> None:
        _import_bitsandbytes()
        if default_dtype is not None:
            default_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.default_dtype = default_dtype

        if dtype is None:
            # try to be smart about the default selection
            if mode.startswith("int8"):
                dtype = torch.float16
            else:
                dtype = (
                    torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                )
        self.dtype = dtype


    @override
    def tensor_init_context(self) -> ContextManager:
        """Controls how tensors get created (device, dtype)."""
        # return nullcontext()
        return _DtypeContextManager(self.default_dtype)


    @override
    def forward_context(self) -> ContextManager:
        """A contextmanager for managing model forward/training_step/evaluation_step/predict_step."""
        # return nullcontext()
        return _DtypeContextManager(self.default_dtype)

    @override
    def convert_input(self, data: Any) -> Any:
        """Convert model inputs (forward) to the floating point precision type of this plugin.

        This is a no-op in the base precision plugin, since we assume the data already has the desired type (default is
        torch.float32).

        """
        # return data
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=self.dtype)

    @override
    def convert_output(self, data: Any) -> Any:
        """Convert outputs to the floating point precision type expected after model's forward.

        This is a no-op in the base precision plugin, since we assume the data already has the desired type (default is
        torch.float32).

        """
        # return data
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=torch.get_default_dtype())
