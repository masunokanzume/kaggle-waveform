RUN_VALID = True
RUN_TEST  = True

import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise RuntimeError("Requires >= 2 GPUs with CUDA enabled.")

import monai
