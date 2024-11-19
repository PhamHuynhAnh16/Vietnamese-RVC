import os
import sys
import torch
import inspect
import warnings
import functools

from pathlib import Path
from diffq import restore_quantized_state

now_dir = os.getcwd()
sys.path.append(now_dir)

from main.configs.config import Config

translations = Config().translations


def load_model(path_or_package, strict=False):
    if isinstance(path_or_package, dict): package = path_or_package
    elif isinstance(path_or_package, (str, Path)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            path = path_or_package
            package = torch.load(path, map_location="cpu")
    else: raise ValueError(f"{translations['type_not_valid']} {path_or_package}.")


    klass = package["klass"]
    args = package["args"]
    kwargs = package["kwargs"]


    if strict: model = klass(*args, **kwargs)
    else:
        sig = inspect.signature(klass)

        for key in list(kwargs):
            if key not in sig.parameters:
                warnings.warn(translations["del_parameter"] + key)
                
                del kwargs[key]

        model = klass(*args, **kwargs)

    state = package["state"]

    set_state(model, state)

    return model


def set_state(model, state, quantizer=None):
    if state.get("__quantized"):
        if quantizer is not None: quantizer.restore_quantized_state(model, state["quantized"])
        else: restore_quantized_state(model, state)
    else: model.load_state_dict(state)

    return state


def capture_init(init):
    @functools.wraps(init)

    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__