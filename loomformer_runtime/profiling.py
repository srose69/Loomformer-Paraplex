import contextlib

import torch


def profile_region(name: str):
    if torch.compiler.is_compiling():
        return contextlib.nullcontext()
    return torch.profiler.record_function(name)
