# Copyright (c) ModelScope Contributors. All rights reserved.
import os

if os.getenv("ACCELERATOR_BACKEND") == "musa":
    import musa_patch

if __name__ == '__main__':
    os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
    from swift.megatron import megatron_pretrain_main
    megatron_pretrain_main()
