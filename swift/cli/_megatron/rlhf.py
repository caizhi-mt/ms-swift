# Copyright (c) ModelScope Contributors. All rights reserved.
import os

if __name__ == '__main__':
    from swift.cli.utils import try_enable_torchada
    try_enable_torchada()
    os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
    from swift.megatron import megatron_rlhf_main
    megatron_rlhf_main()
