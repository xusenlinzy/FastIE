import platform

import accelerate
import datasets
import torch
import transformers
from transformers.utils import is_torch_cuda_available

VERSION = "2.0.0.dev0"


def print_env() -> None:
    info = {
        "`fastie` version": VERSION,
        "Platform": platform.platform(),
        "Python version": platform.python_version(),
        "PyTorch version": torch.__version__,
        "Transformers version": transformers.__version__,
        "Datasets version": datasets.__version__,
        "Accelerate version": accelerate.__version__,
    }

    if is_torch_cuda_available():
        info["PyTorch version"] += " (GPU)"
        info["GPU type"] = torch.cuda.get_device_name()

    print("\n" + "\n".join(["- {}: {}".format(key, value) for key, value in info.items()]) + "\n")
