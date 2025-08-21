# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from vllm.model_executor.layers.vocab_parallel_embedding import (
        DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from .utils import PPMissingLayer, maybe_offload_to_cpu
from vllm.distributed import get_pp_group

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.distributions.normal import Normal
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)



from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.activations import ACT2FN

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import LossKwargs, auto_docstring, can_return_tuple, is_torch_flex_attn_available, logging
""" PyTorch LLaMA-MoE model."""
from dataclasses import dataclass
from typing import Optional, Tuple, Protocol
import math
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaPreTrainedModel,
)
from .llama import *
from .llama_moeconfig import LlamaMoEConfig
from transformers.utils import ModelOutput, logging


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from transformers.integrations.flex_attention import make_flex_block_causal_mask

from transformers.integrations import use_kernel_forward_from_hub


logger = logging.get_logger(__name__)


_ORIG_LLAMAMODEL_INIT = LlamaModel.__init__



class LayerFn(Protocol):

    def __call__(self, prefix: str) -> torch.nn.Module:
        ...

def make_moe_layers(
    num_hidden_layers: int,
    layer_fn: LayerFn,
    prefix: str,
) -> Tuple[int, int, torch.nn.ModuleList]:
    """Make a list of layers with the given layer function, taking
    pipeline parallelism into account.
    """
    from vllm.distributed.parallel_state import get_pp_group
    from vllm.distributed.utils import get_pp_indices
    start_layer, end_layer = get_pp_indices(num_hidden_layers,
                                            get_pp_group().rank_in_group,
                                            get_pp_group().world_size)

    modules = torch.nn.ModuleList(
        [PPMissingLayer() for _ in range(start_layer)] + [
            maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{idx}", layer_index=idx))
            for idx in range(start_layer, end_layer)
        ] + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)])
    return start_layer, end_layer, modules

class LinearGLUExperts(nn.Module):
    """
    Modified from transformers.models.llama.modeling_llama.LlamaMLP
    """

    __constants__ = [
        "bias",
        "in_features",
        "hidden_features",
        "out_features",
        "hidden_act",
        "num_experts",
        "size_experts",
    ]

    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        hidden_act,
        num_experts,
        size_experts=None,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(LinearGLUExperts, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.hidden_act = hidden_act
        self.num_experts = num_experts

        if size_experts is None:
            # all experts share the same number of hidden neurons
            assert hidden_features % num_experts == 0
            size_per_expert = hidden_features // num_experts
            size_experts = [size_per_expert for _ in range(num_experts)]
        else:
            # use specified expert sizes
            assert (
                len(size_experts) == num_experts
                and sum(size_experts) == hidden_features
            )
        self.size_experts = size_experts

        self.act_fn = ACT2FN[hidden_act]

        self.weight_gate = nn.ParameterList()
        self.weight_up = nn.ParameterList()
        self.weight_down = nn.ParameterList()

        for i in range(num_experts):
            # this matrix will be transposed when performing linear forwarding
            this_expert_weight_gate = nn.Parameter(
                torch.empty((size_experts[i], in_features), **factory_kwargs)
            )
            # this matrix will be transposed when performing linear forwarding
            this_expert_weight_up = nn.Parameter(
                torch.empty((size_experts[i], in_features), **factory_kwargs)
            )
            # this matrix will be transposed when performing linear forwarding
            this_expert_weight_down = nn.Parameter(
                torch.empty((out_features, size_experts[i]), **factory_kwargs)
            )
            self.weight_gate.append(this_expert_weight_gate)
            self.weight_up.append(this_expert_weight_up)
            self.weight_down.append(this_expert_weight_down)

        if bias:
            self.bias_gate = nn.ParameterList()
            self.bias_up = nn.ParameterList()
            self.bias_down = nn.ParameterList()

            for i in range(num_experts):
                this_expert_bias_gate = nn.Parameter(
                    torch.empty((size_experts[i],), **factory_kwargs)
                )
                this_expert_bias_up = nn.Parameter(
                    torch.empty((size_experts[i],), **factory_kwargs)
                )
                this_expert_bias_down = nn.Parameter(
                    torch.empty((out_features,), **factory_kwargs)
                )
                self.bias_gate.append(this_expert_bias_gate)
                self.bias_up.append(this_expert_bias_up)
                self.bias_down.append(this_expert_bias_down)
        else:
            self.register_parameter("bias_gate", None)
            self.register_parameter("bias_up", None)
            self.register_parameter("bias_down", None)

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.weight_gate[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.weight_up[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.weight_down[i], a=math.sqrt(5))
            if self.bias_gate is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_gate[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias_gate[i], -bound, bound)
            if self.bias_up is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_up[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias_up[i], -bound, bound)
            if self.bias_down is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_down[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias_down[i], -bound, bound)

    def forward(self, input, i):
        gate = self.act_fn(
            F.linear(
                input,
                self.weight_gate[i],
                self.bias_gate[i] if self.bias_gate is not None else None,
            )
        )
        up = F.linear(
            input,
            self.weight_up[i],
            self.bias_up[i] if self.bias_up is not None else None,
        )
        down = F.linear(
            gate * up,
            self.weight_down[i],
            self.bias_down[i] if self.bias_down is not None else None,
        )
        return down

    def extra_repr(self):
        return (
            "in_features={}, hidden_features={}, out_features={}, hidden_act={},"
            " num_experts={}, size_experts={}, bias={}".format(
                self.in_features,
                self.hidden_features,
                self.out_features,
                self.hidden_act,
                self.num_experts,
                self.size_experts,
                self.bias_gate is not None,
            )
        )



class UniversalCalculator(nn.Module):
    def __init__(
        self,
        experts: LinearGLUExperts,
        multiply_gate_scores=True,
        score_scale_factor=1.0,
    ):
        super(UniversalCalculator, self).__init__()
        self.experts = experts
        # TODO (zhutong): use vmap to boost the training efficiency
        # self.experts_vmap = torch.vmap(self.experts)
        self.multiply_gate_scores = multiply_gate_scores
        self.score_scale_factor = score_scale_factor
        self.num_experts = experts.num_experts
        self.mlp_norm = None

    def reset_experts(self):
        self.experts.reset_parameters()

    def forward(
        self, x, topK_indices, topK_scores, expert_batch_size=None, **kwargs
    ):
        batch_size = topK_indices.size(0)  # topK_indices: (bsz*seq_len, num_selects)
        num_selects = topK_indices.size(1)
        topK_indices = topK_indices.flatten()  # shape(batch_size*num_selects)
        topK_scores = topK_scores.flatten()  # shape(batch_size*num_selects)
        batch_indices = torch.arange(
            batch_size, device=topK_scores.device
        ).repeat_interleave(num_selects)

        _, index_sorted_topK_indices = topK_indices.sort(0)

        sorted_topK_scores = topK_scores.index_select(0, index_sorted_topK_indices)
        sorted_batch_indices = batch_indices.index_select(0, index_sorted_topK_indices)
        if expert_batch_size is None:
            expert_batch_size = topK_indices.bincount(
                minlength=self.num_experts
            ).tolist()

            # unique_indices, counts = torch.unique(topK_indices, return_counts=True)
            # expert_batch_size = torch.zeros(num_experts, device=topK_indices.device)
            # expert_batch_size[unique_indices] = counts.float()

        sorted_x = x.index_select(0, sorted_batch_indices)
        split_x = torch.split(sorted_x, expert_batch_size, dim=0)

        expert_outputs = [
            self.experts(split_x[i], i)
            for i in range(self.num_experts)
            if split_x[i].shape[0] > 0
        ]

        # (bsz*seq_len*num_selects, hidden_size)
        cat_expert_outputs = torch.cat(expert_outputs, 0)
        output_dim = cat_expert_outputs.size(1)
        if self.multiply_gate_scores:
            if self.mlp_norm is None:
                cat_expert_outputs = torch.mul(
                    cat_expert_outputs,
                    sorted_topK_scores.reshape(-1, 1) * self.score_scale_factor,
                )
                # cat_expert_outputs = torch.mul(cat_expert_outputs, sorted_topK_scores.reshape(-1, 1) * 1.0)
            else:
                cat_expert_outputs = torch.mul(
                    cat_expert_outputs, sorted_topK_scores.reshape(-1, 1)
                )
                cat_expert_outputs = self.mlp_norm(cat_expert_outputs)

        zeros = torch.zeros(
            (batch_size, output_dim),
            device=cat_expert_outputs.device,
            dtype=cat_expert_outputs.dtype,
        )
        y = zeros.index_add(0, sorted_batch_indices, cat_expert_outputs)

        return y


class TopKBalancedNoisyGate(nn.Module):
    def __init__(
        self,
        input_size,
        num_experts,
        num_selects,
        gate_network="mlp",
        use_softmax=True,
        use_balance=True,
        balance_loss_weight=1e-2,
        add_noise=True,
        noise_epsilon=1e-2,
    ):
        super(TopKBalancedNoisyGate, self).__init__()
        assert num_selects <= num_experts
        self.input_size = input_size
        self.num_experts = num_experts
        self.num_selects = num_selects

        self.gate_network_type = gate_network
        self.gate_network = self.get_gate_network(gate_network, input_size, num_experts)

        self.use_softmax = use_softmax
        self.softmax = nn.Softmax(1)

        self.use_balance = use_balance
        self.balance_loss_weight = balance_loss_weight

        # add_noise
        self.add_noise = add_noise
        self.noise_epsilon = noise_epsilon
        self.warned = False
        if self.add_noise:
            self.weight_noise = nn.Linear(input_size, num_experts, bias=False)
            self.weight_noise.weight.data = torch.zeros(
                (num_experts, input_size),
                requires_grad=True,
                device=self.weight_noise.weight.data.device,
                dtype=self.weight_noise.weight.data.dtype,
            )
            self.mean = 0.0
            self.std = 1.0
            self.normal = Normal(self.mean, self.std)
            self.softplus = nn.Softplus()

        self.reset_parameters()

    def reset_gate_network(self):
        self.gate_network = self.get_gate_network(
            self.gate_network_type, self.input_size, self.num_experts
        )

    def reset_parameters(self):
        if self.add_noise:
            nn.init.zeros_(self.weight_noise.weight)

    def get_gate_network(self, gate_network_type, input_size, num_experts):
        gate_network_type = gate_network_type.lower()

        if gate_network_type == "linear":
            gate_network = nn.Linear(input_size, num_experts, bias=False)
            nn.init.zeros_(gate_network.weight)
        elif gate_network_type == "mlp":
            gate_network = torch.nn.Sequential(
                torch.nn.Linear(input_size, num_experts, bias=False),
                torch.nn.Tanh(),
                torch.nn.Linear(num_experts, num_experts, bias=False),
            )
        else:
            raise ValueError(f"Unexpected gate network type: {gate_network_type}.")

        return gate_network

    def cv_squared(self, x, eps=1e-10):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.s
        """
        if x.shape[0] == 1:
            return torch.tensor(0.0, device=x.device)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x):
        logits_gate = self.gate_network(x)
        
        logits = logits_gate

        top_logits, top_indices = logits.topk(
            min(self.num_selects + 1, self.num_experts), dim=1
        )  # select the top (k+1) experts
        top_k_logits = top_logits[:, : self.num_selects]
        top_k_indices = top_indices[:, : self.num_selects]
        top_k_scores = (
            self.softmax(top_k_logits.to(torch.float32))
            if self.use_softmax
            else top_k_logits
        )
        top_k_scores = top_k_scores.to(logits.dtype)

        zeros = torch.zeros_like(logits, device=logits.device)
        scores_filtered = zeros.scatter(
            dim=1, index=top_k_indices, src=top_k_scores
        )  # shape(batch_size, num_experts)
        importance = scores_filtered.sum(0)  # shape(num_experts)

        
        load = (scores_filtered > 0).sum(0)

        if self.use_balance:
            balance_loss = self.cv_squared(importance) + self.cv_squared(load)
            balance_loss *= self.balance_loss_weight
        else:
            balance_loss = torch.tensor(-100.0, device=x.device)

        return {
            "topK_indices": top_k_indices,
            "topK_scores": top_k_scores,
            "balance_loss": balance_loss,
            "load": load,
            "importance": importance,
        }

class BaseMoELayer(nn.Module):
    def __init__(self):
        super(BaseMoELayer, self).__init__()
        self.gate = None
        self.calculator = None

    
        
    def forward(self, x):
        original_shape = x.shape[:-1]
        x = x.reshape(-1, self.input_size)
        gate_outputs: dict = self.gate(x)
        y = self.calculator(x, **gate_outputs)
        y = y.reshape(original_shape + (self.output_size,))
        # shape(batch_size, seq_len, output_size)
        return y

    
    



class LinearGLUMoELayer(BaseMoELayer):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        hidden_act,
        num_experts,
        num_selects,
        size_experts=None,
        bias=True,
        **kwargs,
    ):
        # fmt: off
        super(LinearGLUMoELayer, self).__init__()
        assert (num_selects <= num_experts)  # 选择数量大于专家数量，报错
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_act = hidden_act
        self.num_experts = num_experts
        self.num_selects = num_selects
        self.size_experts = size_experts
        self.bias = bias

        experts = LinearGLUExperts(
            input_size,
            hidden_size,
            output_size,
            hidden_act,
            num_experts,
            size_experts=size_experts,
            bias=bias
        )

        self.gate = TopKBalancedNoisyGate(
            self.input_size,
            self.num_experts,
            self.num_selects,
            gate_network=kwargs.get("gate_network", "mlp"),
            use_softmax=kwargs.get("gate_use_softmax", True),
            use_balance=kwargs.get("gate_use_balance", True),
            balance_loss_weight=kwargs.get("gate_balance_loss_weight", 1e-2),
            add_noise=kwargs.get("gate_add_noise", True),
            noise_epsilon=kwargs.get("gate_noise_epsilon", 1e-2),
        )

        # create calculator
        self.calculator = UniversalCalculator(
            experts,
            multiply_gate_scores=kwargs.get("multiply_gate_scores", True),
            score_scale_factor=kwargs.get("score_scale_factor", 1.0),
        )






class LlamaMoEDecoderLayer(LlamaDecoderLayer):
    def __init__(self,
                layer_index,
                config:LlamaMoEConfig,
                cache_config: Optional[CacheConfig] = None,
                quant_config: Optional[QuantizationConfig] = None,
                prefix: str = ""):
        super().__init__(config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix)



        gating_config = {
            # all gates
            "gate_type": config.gate_type,
            "gate_network": config.gate_network,
            "gate_use_softmax": config.gate_use_softmax,
            "gate_use_balance": config.gate_use_balance,
            "gate_balance_loss_weight": config.gate_balance_loss_weight,
            "gate_add_noise": config.gate_add_noise,
            # TopKBalancedNoisyGate
            "gate_noise_epsilon": config.gate_noise_epsilon,
        }
        calculator_config = {
            # all calculators
            "calculator_type": config.calculator_type,
            "multiply_gate_scores": config.multiply_gate_scores,
            "score_scale_factor": (
                config.score_scale_factor[layer_index]
                if isinstance(config.score_scale_factor, list)
                else config.score_scale_factor
            ),
            "add_weight_norm": config.add_weight_norm,
            # SwitchDropTokenCalculator
            "drop_tokens": config.drop_tokens,
            "dropped_padding": config.dropped_padding,
            "capacity_factor": config.capacity_factor,
        }
        self.mlp = LinearGLUMoELayer(
            input_size=self.hidden_size,
            hidden_size=config.intermediate_size,
            output_size=self.hidden_size,
            hidden_act=config.hidden_act,
            num_experts=config.num_experts,
            num_selects=config.num_selects,
            size_experts=(
                config.size_experts[layer_index]
                if config.size_experts is not None
                else None
            ),
            bias=False,
            **gating_config,
            **calculator_config,
        )





def _patched_llama_model_init(self, *, vllm_config, prefix="", layer_type: type[nn.Module] = LlamaMoEDecoderLayer):
    # 调用 nn.Module 的 init，保证能注册子模块
    nn.Module.__init__(self)

    config = vllm_config.model_config.hf_config
    cache_config = vllm_config.cache_config
    quant_config = vllm_config.quant_config
    lora_config = vllm_config.lora_config

    self.config = config
    self.quant_config = quant_config
    lora_vocab = (lora_config.lora_extra_vocab_size *
                    (lora_config.max_loras or 1)) if lora_config else 0
    self.vocab_size = config.vocab_size + lora_vocab
    self.org_vocab_size = config.vocab_size
    if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                        and get_pp_group().is_last_rank):
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=quant_config,
        )
    else:
        self.embed_tokens = PPMissingLayer()

    self._init_layers(vllm_config, prefix, layer_type)
    if get_pp_group().is_last_rank:
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    else:
        self.norm = PPMissingLayer()

    self.aux_hidden_state_layers: tuple[int] = tuple()

    self.make_empty_intermediate_tensors = (
        make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size))

LlamaModel.__init__ = _patched_llama_model_init



# # @auto_docstring
class LlamaMoEModel(LlamaModel):
    def _init_layers(self, vllm_config, prefix, layer_type):
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.start_layer, self.end_layer, self.layers = make_moe_layers(
            config.num_hidden_layers,
            lambda prefix, layer_index: LlamaMoEDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
                layer_index=layer_index,
            ),
            prefix=f"{prefix}.layers",
        )
        

    def set_input_embeddings(self, value):
        self.embed_tokens = value




# @auto_docstring
class LlamaMoEForCausalLM(LlamaForCausalLM):
    def __init__(self,
                 vllm_config: VllmConfig,
                 prefix: str = ""):
        super().__init__(vllm_config=vllm_config, layer_type=LlamaMoEDecoderLayer)
    def _init_model(self,
                    vllm_config: VllmConfig,
                    prefix: str = "",
                    layer_type: type[nn.Module] = LlamaMoEDecoderLayer):
        print("init llama-moe")
        return LlamaMoEModel(vllm_config=vllm_config,
                          prefix=prefix
                          )

    

    



