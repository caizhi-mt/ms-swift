# Copyright (c) ModelScope Contributors. All rights reserved.
from copy import deepcopy
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.ssm.gated_delta_net import GatedDeltaNetSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from typing import Optional

from swift.megatron.utils import get_local_layer_specs
from swift.model import ModelType
from swift.utils import get_env_args
from ..constant import MegatronModelType
from ..modules import GatedDeltaNet
from ..register import MegatronModelMeta, register_megatron_model
from .qwen3_5 import Qwen3_5Bridge as BaseQwen3_5Bridge
from .qwen3_5 import Qwen3_5Loader as BaseQwen3_5Loader
from .qwen3_5 import Qwen3_5Vit
from ..gpts.qwen3_next import Qwen3NextRMSNorm, Qwen3NextSelfAttention, TEColumnParallelLinear, mcore_013


class Qwen3_5Bridge(BaseQwen3_5Bridge):
    hf_mtp_prefix = 'mtp.layers'

    def _set_layer_attn(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool):
        is_linear_attention = self.config.linear_attention_freq[layer_idx]
        if is_linear_attention:
            mg_attn = None if mg_layer is None else mg_layer.self_attention
            hf_state_dict.update(
                self._set_linear_attn_state(mg_attn, hf_state_dict, 'linear_attn.', layer_idx, to_mcore))
            self._set_state_dict(mg_layer, 'input_layernorm.weight', hf_state_dict,
                                 'input_layernorm.weight', to_mcore)
            return hf_state_dict
        return super()._set_layer_attn(mg_layer, hf_state_dict, layer_idx, to_mcore)


class Qwen3_5Loader(BaseQwen3_5Loader):
    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        config = self.config
        args = self.args
        config.hetereogenous_dist_checkpoint = True
        config.hidden_act = 'silu'
        config.rms_norm_eps = config.layernorm_epsilon
        config.dtype = args.torch_dtype

        layer_norm_impl = Qwen3NextRMSNorm
        kwargs = {'use_kitchen': config.use_kitchen} if mcore_013 else {}
        moe_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            **kwargs,
        )
        backend = TESpecProvider()
        gdn_attn_spec = ModuleSpec(
            module=GatedDeltaNet,
            submodules=GatedDeltaNetSubmodules(
                in_proj=backend.column_parallel_linear(),
                out_norm=backend.layer_norm(rms_norm=config.normalization == 'RMSNorm', for_qk=False),
                out_proj=backend.row_parallel_linear(),
            ),
        )

        layer_specs = []
        for is_linear_attention in self.config.linear_attention_freq:
            layer_spec = deepcopy(moe_layer_spec)
            if is_linear_attention:
                layer_spec.submodules.self_attention = deepcopy(gdn_attn_spec)
            else:
                layer_spec.submodules.self_attention.module = Qwen3NextSelfAttention
                layer_spec.submodules.self_attention.submodules.linear_qkv = TEColumnParallelLinear
            layer_spec.submodules.input_layernorm = layer_norm_impl
            if hasattr(layer_spec.submodules, 'pre_mlp_layernorm'):
                layer_spec.submodules.pre_mlp_layernorm = layer_norm_impl
            if config.hf_model_type == 'qwen3_5':
                layer_spec.submodules.mlp.submodules.linear_fc1 = TEColumnParallelLinear
            if not is_linear_attention:
                if hasattr(layer_spec.submodules.self_attention.submodules, 'q_layernorm'):
                    layer_spec.submodules.self_attention.submodules.q_layernorm = layer_norm_impl
                if hasattr(layer_spec.submodules.self_attention.submodules, 'k_layernorm'):
                    layer_spec.submodules.self_attention.submodules.k_layernorm = layer_norm_impl
            layer_specs.append(layer_spec)

        local_layer_specs = get_local_layer_specs(config, layer_specs, vp_stage=vp_stage)
        return TransformerBlockSubmodules(layer_specs=local_layer_specs, layer_norm=layer_norm_impl)

    def build_model(
        self,
        pre_process=True,
        post_process=True,
        vp_stage: Optional[int] = None,
    ):
        model = super().build_model(pre_process, post_process, vp_stage)
        lm_model = model.language_model if hasattr(model, 'language_model') else model
        for layer in lm_model.decoder.layers:
            if hasattr(layer.self_attention, 'out_norm'):
                assert hasattr(layer.self_attention.out_norm, 'zero_centered_gamma')
                layer.self_attention.out_norm.zero_centered_gamma = False
        return model


use_mcore_gdn = get_env_args('SWIFT_USE_MCORE_GDN', bool, False)

if use_mcore_gdn:
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
