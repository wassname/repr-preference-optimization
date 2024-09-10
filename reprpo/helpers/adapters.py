from contextlib import contextmanager
from peft import PeftModel

@contextmanager
def set_adapter(model: PeftModel, adapter_name: str = None):
    old_adapter_name = model.active_adapter
    try:
        if adapter_name is not None:
            model.set_adapter(adapter_name)
            yield model
        else:
            with model.disable_adapter():
                yield model
    finally:
        model.set_adapter(old_adapter_name)
