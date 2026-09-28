import os

# On a GPU-less machine, fall back to Triton's interpreter so the kernel
# tests still check correctness (pure CPU). Interpreter timings mean
# nothing, so benchmark numbers always come from a real GPU. This must run
# before any test imports triton, and pytest loads conftest at exactly the
# right moment for that.
try:
    import torch

    if "TRITON_INTERPRET" not in os.environ and not torch.cuda.is_available():
        os.environ["TRITON_INTERPRET"] = "1"
except ImportError:
    pass
