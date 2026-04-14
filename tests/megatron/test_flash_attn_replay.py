import os
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


def _is_torch_musa_available() -> bool:
    musa = getattr(torch, 'musa', None)
    if musa is None or not hasattr(musa, 'is_available'):
        return False
    try:
        return musa.is_available()
    except Exception:
        return False


def _get_accelerator_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    if _is_torch_musa_available():
        return torch.device('musa:0')
    return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_case_path() -> Path:
    case_path = os.getenv('SWIFT_FLASH_ATTN_CASE_PATH')
    if case_path:
        return Path(case_path).expanduser().resolve()

    matches = sorted(_repo_root().glob('logs/**/flash_attn_debug_cases/flash_attn_case_layer*_rank*_*.pt'))
    return matches[-1].resolve() if matches else None


def _expand_kv_heads(tensor: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    num_kv_heads = tensor.shape[1]
    if num_kv_heads == num_query_heads:
        return tensor
    assert num_query_heads % num_kv_heads == 0, (num_query_heads, num_kv_heads)
    repeat_factor = num_query_heads // num_kv_heads
    return tensor.repeat_interleave(repeat_factor, dim=1)


def _build_reference_output(case: dict) -> torch.Tensor:
    packed_seq_params = case['packed_seq_params']
    query = case['query'].float()
    key = case['key'].float()
    value = case['value'].float()
    cu_seqlens_q = packed_seq_params['cu_seqlens_q'].to(dtype=torch.int64)
    cu_seqlens_kv = packed_seq_params['cu_seqlens_kv'].to(dtype=torch.int64)
    causal = case['metadata']['attn_mask_type'] == 'causal'

    outputs = []
    num_query_heads = query.shape[1]
    for i in range(cu_seqlens_q.numel() - 1):
        q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        k_start, k_end = cu_seqlens_kv[i].item(), cu_seqlens_kv[i + 1].item()

        query_i = query[q_start:q_end].transpose(0, 1).unsqueeze(0).contiguous()
        key_i = _expand_kv_heads(key[k_start:k_end], num_query_heads).transpose(0, 1).unsqueeze(0).contiguous()
        value_i = _expand_kv_heads(value[k_start:k_end], num_query_heads).transpose(0, 1).unsqueeze(0).contiguous()

        out_i = F.scaled_dot_product_attention(query_i, key_i, value_i, dropout_p=0.0, is_causal=causal)
        outputs.append(out_i.squeeze(0).transpose(0, 1).contiguous())

    return torch.cat(outputs, dim=0)


def _build_varlen_cu_seqlens(total_tokens: int, device: torch.device) -> tuple[torch.Tensor, int]:
    # Real replay case lengths are roughly:
    # [198, 198, 198, 197 x5, 196 x7, 195 x3, 194 x2, 192 x2, 191 x3,
    #  190 x5, 189 x4, 188 x2, 187 x4, 186 x2, 105]
    # Use the same descending packing style for synthetic varlen cases so they
    # exercise a distribution closer to the real training path.
    base_lengths = [
        198, 198, 198,
        197, 197, 197, 197, 197,
        196, 196, 196, 196, 196, 196, 196,
        195, 195, 195,
        194, 194,
        192, 192,
        191, 191, 191,
        190, 190, 190, 190, 190,
        189, 189, 189, 189,
        188, 188,
        187, 187, 187, 187,
        186, 186,
    ]
    lengths = []
    remaining = total_tokens
    idx = 0
    while remaining > 0:
        candidate = base_lengths[idx % len(base_lengths)]
        if remaining <= candidate + 105:
            lengths.append(remaining)
            break
        lengths.append(candidate)
        remaining -= candidate
        idx += 1

    cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(dim=0)
    return cu_seqlens, max(lengths)


def _make_synthetic_tensor(shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    tensor = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.5
    return tensor.to(device=device, dtype=dtype)


class TestFlashAttnReplay(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.case_path = _resolve_case_path()
        cls.device = _get_accelerator_device()
        if cls.device is None:
            raise unittest.SkipTest('FlashAttention replay test requires CUDA or MUSA.')

        try:
            from flash_attn import flash_attn_varlen_func  # noqa: F401
        except Exception as e:
            raise unittest.SkipTest(f'flash_attn is unavailable: {e}')

    def _load_case(self):
        if self.case_path is None or not self.case_path.exists():
            self.skipTest(
                'No replay case found. Set `SWIFT_FLASH_ATTN_CASE_PATH` or dump a case under '
                '`logs/**/flash_attn_debug_cases/*.pt`.')
        return torch.load(self.case_path, map_location='cpu')

    def _build_synthetic_case(
        self,
        *,
        total_tokens: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> dict:
        cu_seqlens, max_seqlen = _build_varlen_cu_seqlens(total_tokens, self.device)
        base_seed = total_tokens * 1000 + num_query_heads * 100 + num_kv_heads * 10 + head_dim
        query = _make_synthetic_tensor(
            (total_tokens, num_query_heads, head_dim), dtype=dtype, device=self.device, seed=base_seed + 1)
        key = _make_synthetic_tensor(
            (total_tokens, num_kv_heads, head_dim), dtype=dtype, device=self.device, seed=base_seed + 2)
        value = _make_synthetic_tensor(
            (total_tokens, num_kv_heads, head_dim), dtype=dtype, device=self.device, seed=base_seed + 3)
        return {
            'metadata': {
                'branch': 'synthetic_static_core_attention',
                'attn_mask_type': 'causal',
                'synthetic_total_tokens': total_tokens,
                'synthetic_num_samples': cu_seqlens.numel() - 1,
                'synthetic_max_seqlen': max_seqlen,
                'synthetic_kv_heads': num_kv_heads,
                'synthetic_seed_base': base_seed,
            },
            'query': query,
            'key': key,
            'value': value,
            'packed_seq_params': {
                'qkv_format': 'thd',
                'cu_seqlens_q': cu_seqlens,
                'cu_seqlens_kv': cu_seqlens.clone(),
                'max_seqlen_q': max_seqlen,
                'max_seqlen_kv': max_seqlen,
                'num_samples': cu_seqlens.numel() - 1,
            },
        }

    def _flash_attn_forward(self, case: dict) -> torch.Tensor:
        from flash_attn import flash_attn_varlen_func

        packed_seq_params = case['packed_seq_params']
        query = case['query'].to(device=self.device, dtype=case['query'].dtype)
        key = case['key'].to(device=self.device, dtype=case['key'].dtype)
        value = case['value'].to(device=self.device, dtype=case['value'].dtype)
        cu_seqlens_q = packed_seq_params['cu_seqlens_q'].to(device=self.device, dtype=torch.int32)
        cu_seqlens_kv = packed_seq_params['cu_seqlens_kv'].to(device=self.device, dtype=torch.int32)
        causal = case['metadata']['attn_mask_type'] == 'causal'

        with torch.no_grad():
            return flash_attn_varlen_func(
                query,
                key,
                value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_kv,
                max_seqlen_q=int(packed_seq_params['max_seqlen_q']),
                max_seqlen_k=int(packed_seq_params['max_seqlen_kv']),
                dropout_p=0.0,
                causal=causal,
                deterministic=True,
            )

    def test_flash_attn_varlen_replay_matches_reference(self):
        case = self._load_case()
        self.assertEqual(case['packed_seq_params']['qkv_format'], 'thd')
        self.assertIn(case['metadata']['branch'], {'static_core_attention', 'checkpointed_attention'})

        flash_out = self._flash_attn_forward(case)
        ref_out = _build_reference_output(case)

        self.assertEqual(tuple(flash_out.shape), tuple(ref_out.shape))

        flash_out_cpu = flash_out.detach().cpu().float()
        gate = case['gate'].reshape_as(ref_out).float()
        flash_out_gated = flash_out_cpu * torch.sigmoid(gate)
        ref_out_gated = ref_out * torch.sigmoid(gate)

        abs_err = (flash_out_cpu - ref_out).abs()
        gated_abs_err = (flash_out_gated - ref_out_gated).abs()
        max_abs = abs_err.max().item()
        mean_abs = abs_err.mean().item()
        gated_max_abs = gated_abs_err.max().item()
        gated_mean_abs = gated_abs_err.mean().item()

        try:
            torch.testing.assert_close(flash_out_cpu, ref_out, rtol=1e-1, atol=1e-1)
            torch.testing.assert_close(flash_out_gated, ref_out_gated, rtol=1e-1, atol=1e-1)
        except AssertionError as e:
            self.fail(
                f'flash_attn replay mismatch for case `{self.case_path}`. '
                f'max_abs={max_abs:.6f}, mean_abs={mean_abs:.6f}, '
                f'gated_max_abs={gated_max_abs:.6f}, gated_mean_abs={gated_mean_abs:.6f}\n{e}')

    def _run_synthetic_large_shape_case(self, *, total_tokens: int, head_dim: int):
        case = self._build_synthetic_case(
            total_tokens=total_tokens,
            num_query_heads=48,
            num_kv_heads=8,
            head_dim=head_dim,
        )

        try:
            out = self._flash_attn_forward(case)
        except RuntimeError as e:
            self.fail(
                'flash_attn varlen synthetic case failed: '
                f'total_tokens={total_tokens}, head_dim={head_dim}, '
                f'num_samples={case["packed_seq_params"]["num_samples"]}, '
                f'max_seqlen={case["packed_seq_params"]["max_seqlen_q"]}, error={e}')

        self.assertEqual(tuple(out.shape), tuple(case['query'].shape))
        self.assertEqual(out.dtype, case['query'].dtype)


_SYNTHETIC_TOTAL_TOKENS = [15818, 20649, 32662, 32743, 32768]
_SYNTHETIC_HEAD_DIMS = [128, 256]


def _make_synthetic_large_shape_test(total_tokens: int, head_dim: int):

    def _test(self):
        self._run_synthetic_large_shape_case(total_tokens=total_tokens, head_dim=head_dim)

    _test.__name__ = f'test_flash_attn_varlen_synthetic_tokens_{total_tokens}_hd_{head_dim}'
    return _test


for _head_dim in _SYNTHETIC_HEAD_DIMS:
    for _total_tokens in _SYNTHETIC_TOTAL_TOKENS:
        setattr(
            TestFlashAttnReplay,
            f'test_flash_attn_varlen_synthetic_tokens_{_total_tokens}_hd_{_head_dim}',
            _make_synthetic_large_shape_test(_total_tokens, _head_dim),
        )


if __name__ == '__main__':
    unittest.main()
