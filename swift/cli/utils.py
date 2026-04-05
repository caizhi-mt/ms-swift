# Copyright (c) ModelScope Contributors. All rights reserved.
import os


def try_enable_torchada():
    enable = os.environ.get('SWIFT_ENABLE_TORCHADA', '0').lower() in {'1', 'true', 'yes', 'on'}
    if not enable:
        return
    try:
        import torchada  # noqa: F401
        print('[swift] torchada enabled.', flush=True)
    except ImportError:
        print('[swift] SWIFT_ENABLE_TORCHADA is set but torchada is not installed. '
              'Install it via `pip install torchada`.', flush=True)


def try_enable_musa_patch():
    enable = os.environ.get('ENABLE_MEGATRON_MUSA_PATCH', '0').lower() in {'1', 'true', 'yes', 'on'}
    if not enable:
        return
    if os.environ.get('ACCELERATOR_BACKEND') != 'musa':
        print('[swift] ENABLE_MEGATRON_MUSA_PATCH is set but ACCELERATOR_BACKEND is not `musa`; skipping.',
              flush=True)
        return
    try:
        import musa_patch  # noqa: F401
        print('[swift] musa_patch enabled.', flush=True)
    except Exception as e:
        print(f'[swift] Failed to enable musa_patch: {e!r}', flush=True)


def try_use_single_device_mode():
    if os.environ.get('SWIFT_SINGLE_DEVICE_MODE', '0') == '1':
        env_key = 'CUDA_VISIBLE_DEVICES'
        visible_devices = os.environ.get(env_key)
        if not visible_devices:
            env_key = 'MUSA_VISIBLE_DEVICES'
            visible_devices = os.environ.get(env_key)
        local_rank = os.environ.get('LOCAL_RANK')
        if local_rank is None or not visible_devices:
            return
        visible_devices = visible_devices.split(',')
        visible_device = visible_devices[int(local_rank)]
        os.environ[env_key] = str(visible_device)
        os.environ['LOCAL_RANK'] = '0'
