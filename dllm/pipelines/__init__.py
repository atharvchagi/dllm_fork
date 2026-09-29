"""Load a pipeline on first access; import one directly to run its entrypoint."""

from importlib import import_module


__all__ = [
    "a2d",
    "bert",
    "dream",
    "editflow",
    "fastdllm",
    "lddm_u",
    "llada",
    "llada2",
]


def __getattr__(name):
    """Import a requested pipeline without importing every model family."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
