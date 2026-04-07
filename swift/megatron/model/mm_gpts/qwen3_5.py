# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from megatron.core.extensions.transformer_engine import _get_extra_te_kwargs
from megatron.core.models.huggingface import HuggingFaceModule as _HuggingFaceModule
from megatron.core.tensor_parallel import (gather_from_sequence_parallel_region,
                                           reduce_scatter_to_sequence_parallel_region)
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig

from swift.model import ModelType
from swift.template import Template
from swift.utils import get_env_args
from ..constant import MegatronModelType
from ..gpts.qwen3_next import Qwen3NextBridge, Qwen3NextLoader
from ..register import MegatronModelMeta, register_megatron_model
from .utils import HuggingFaceModule
import os
import torch.nn.functional as F

try:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet as _Qwen3_5MoeGatedDeltaNet, l2norm
except ImportError:
    _Qwen3_5MoeGatedDeltaNet = object

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except ImportError:
    print("请安装 fla 库: pip install flash-linear-attention")


def torch_chunk_gated_delta_rule_patch_musa(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous() for x in (query, key, value, beta, g)
    ]
    #    x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    #decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().float().exp()).tril().to(initial_dtype)
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().to(initial_dtype).unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state.to(initial_dtype)
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp().to(initial_dtype)) @ last_recurrent_state.to(initial_dtype)
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None].to(initial_dtype)).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class Qwen3_5MoeGatedDeltaNet(_HuggingFaceModule, _Qwen3_5MoeGatedDeltaNet):

    def __init__(self, config: TransformerConfig, submodules: SelfAttentionSubmodules, layer_number: int, **kwargs):
        assert config.context_parallel_size == 1, 'Qwen3_5 currently does not support context parallel.'
        assert _Qwen3_5MoeGatedDeltaNet is not object, 'please update the `transformers` version.'
        _Qwen3_5MoeGatedDeltaNet.__init__(self, config, layer_number)
        self.config = config
        extra_kwargs = _get_extra_te_kwargs(config)
        if int(os.getenv("ENABLE_GDN_BF16", 0)):
            self.chunk_gated_delta_rule = chunk_gated_delta_rule
        self.to(dtype=extra_kwargs['params_dtype'], device=extra_kwargs['device'])

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        args = self.config.args
        if args.sequence_parallel and args.tensor_model_parallel_size > 1:
            hidden_states = gather_from_sequence_parallel_region(hidden_states)
        seq_len = hidden_states.shape[0]
        packed_seq_params = kwargs.get('packed_seq_params')
        thd_format = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        # Note: for packed inputs, we do not perform padding_free unpadding.
        # Doing so would allow different sequences to see each other; for efficiency we keep this implementation.
        if thd_format and not args.packing:
            new_hidden_states = hidden_states.new_zeros(
                (packed_seq_params.num_samples, packed_seq_params.max_seqlen_q.item(), hidden_states.shape[-1]))
            attention_mask = hidden_states.new_zeros(
                (packed_seq_params.num_samples, packed_seq_params.max_seqlen_q.item()), dtype=torch.bool)
            cu_seqlens_q = packed_seq_params.cu_seqlens_q
            for i in range(packed_seq_params.num_samples):
                start, end = cu_seqlens_q[i], cu_seqlens_q[i + 1]
                attention_mask[i, :end - start] = True
                new_hidden_states[i, :end - start] = hidden_states[start:end, 0]
            hidden_states = new_hidden_states
        else:
            hidden_states = hidden_states.transpose(0, 1)
            attention_mask = kwargs.get('attention_mask')
            if attention_mask is not None:
                attention_mask = (~attention_mask).sum(dim=(1, 2)) > 0
        res = super().forward(hidden_states=hidden_states, attention_mask=attention_mask)
        if thd_format and not args.packing:
            res = res[attention_mask][:, None]
            res = torch.concat([res, res.new_zeros(seq_len - res.shape[0], 1, res.shape[2])])
        else:
            res = res.transpose(0, 1).contiguous()
        if args.sequence_parallel and args.tensor_model_parallel_size > 1:
            # Quick fix for dropout issue, awaiting ms-swift 4.0 refactor
            res = reduce_scatter_to_sequence_parallel_region(res) / args.tensor_model_parallel_size
        return res, None


class Qwen3_5Vit(HuggingFaceModule):
    module_mapping = {'model.visual': 'visual'}
    _vision_tower = ['visual']
    _aligner = ['visual.merger']

    def __init__(self, config):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel
        super().__init__(config, [Qwen3_5TextModel, Qwen3_5MoeTextModel])

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        return Template._get_inputs_embeds_hf(inputs_embeds, kwargs, self.visual, self.processor, self.hf_config)


class Qwen3_5Bridge(Qwen3NextBridge):
    hf_layers_prefix = 'model.language_model.layers'
    hf_embed_key = 'model.language_model.embed_tokens.weight'
    hf_final_layernorm_key = 'model.language_model.norm.weight'


class Qwen3_5Loader(Qwen3NextLoader):
    gated_delta_net = Qwen3_5MoeGatedDeltaNet


use_mcore_gdn = get_env_args('SWIFT_USE_MCORE_GDN', bool, False)

if not use_mcore_gdn:
    register_megatron_model(
        MegatronModelMeta(
            MegatronModelType.qwen3_5,
            [
                ModelType.qwen3_5,
                ModelType.qwen3_5_moe,
            ],
            bridge_cls=Qwen3_5Bridge,
            visual_cls=Qwen3_5Vit,
            loader=Qwen3_5Loader,
        ))
