# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import time
import torch
from tqdm import tqdm

from swift.megatron.utils import reduce_max_stat_across_model_parallel_group
from swift.utils import JsonlWriter, format_time, get_logger, is_last_rank
from .base import MegatronCallback

logger = get_logger()


class PrintCallback(MegatronCallback):

    def __init__(self, trainer):
        super().__init__(trainer)
        self.training_bar = None
        self.eval_bar = None
        self.jsonl_writer = None
        self.is_write_rank = is_last_rank()

    def on_train_begin(self):
        self.training_bar = tqdm(
            total=self.args.train_iters, dynamic_ncols=True, disable=not self.is_write_rank, desc='Train: ')
        self.start_step = self.state.iteration
        self.training_bar.update(self.state.iteration)
        self.current_step = self.state.iteration
        self.start_time = time.time()
        self.trainer.reset_actual_tokens_per_gpu()
        logging_path = os.path.join(self.args.output_dir, 'logging.jsonl')
        logger.info(f'logging_path: {logging_path}')
        self.jsonl_writer = JsonlWriter(logging_path, enable_async=True, write_on_rank='last')

    def on_train_end(self):
        self.training_bar.close()
        self.training_bar = None

    def on_step_end(self):
        n_step = self.state.iteration - self.current_step
        self.current_step = self.state.iteration
        self.training_bar.update(n_step)

    def on_eval_begin(self):
        self.eval_bar = tqdm(
            total=self.args.eval_iters, dynamic_ncols=True, disable=not self.is_write_rank, desc='Evaluate: ')

    def on_eval_end(self):
        self.eval_bar.close()
        self.eval_bar = None

    def on_eval_step(self):
        self.eval_bar.update()

    def _get_effective_seq_len(self):
        if getattr(self.args, 'packing', False):
            packing_length = getattr(self.args, 'packing_length', None)
            if packing_length is not None:
                return packing_length
        return getattr(self.args, 'seq_length', None) or getattr(self.args, 'max_length', None)

    def _get_world_size(self):
        world_size = getattr(self.args, 'world_size', None)
        if world_size is not None:
            return world_size
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        return None

    def on_log(self, logs):
        state = self.state
        args = self.args
        logs['iteration'] = f'{state.iteration}/{args.train_iters}'
        elapsed = time.time() - self.start_time
        logs['elapsed_time'] = format_time(elapsed)
        n_steps = state.iteration - self.start_step
        train_speed = elapsed / n_steps if n_steps > 0 else 0.0
        logs['remaining_time'] = format_time((args.train_iters - state.iteration) * train_speed)
        memory = reduce_max_stat_across_model_parallel_group(torch.cuda.max_memory_reserved() / 1024**3)
        logs['memory(GiB)'] = round(memory, 2)
        logs['train_speed(s/it)'] = round(train_speed, 6)
        is_eval_log = any(k.startswith('eval_') for k in logs)
        if not is_eval_log and train_speed > 0:
            effective_seq_len = self._get_effective_seq_len()
            num_microbatches = getattr(args, 'num_microbatches', None)
            if effective_seq_len is not None and num_microbatches is not None:
                tokens_per_step_per_gpu_est = args.micro_batch_size * num_microbatches * effective_seq_len
                logs['tokens_per_second_per_gpu_est'] = tokens_per_step_per_gpu_est / train_speed
            world_size = self._get_world_size()
            if effective_seq_len is not None and world_size:
                total_tokens_per_step = args.global_batch_size * effective_seq_len
                logs['tokens_per_second_per_gpu_effective'] = total_tokens_per_step / (train_speed * world_size)
            actual_tokens_per_second = self.trainer.get_actual_tokens_per_second_per_gpu(elapsed)
            if actual_tokens_per_second is not None:
                logs['tokens_per_second_per_gpu_actual'] = actual_tokens_per_second
        logs = {k: round(v, 8) if isinstance(v, float) else v for k, v in logs.items()}
        self.jsonl_writer.append(logs)
        if self.is_write_rank:
            self.training_bar.write(str(logs))
