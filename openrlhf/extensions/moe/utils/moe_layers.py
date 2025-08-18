import warnings
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn
from transformers.utils import ModelOutput

from .moe_calculators import (
    CalculatorOutput,
    SwitchDropTokenCalculator,
    UniformCalculator,
    UniversalCalculator,
)
from .moe_experts import LinearExperts, LinearGLUExperts
from .moe_gates import (
    RandomLearnableGate,
    RandomPlainGate,
    SwitchBalancedGate,
    TopKBalancedNoisyGate,
    UniformLearnableGate,
    UniformPlainGate,
)


@dataclass
class MoEMlpOutput(ModelOutput):
    hidden_states: Optional[torch.FloatTensor] = None
    balance_loss: Optional[torch.FloatTensor] = None
    num_dropped_tokens: Optional[int] = None
    gate_load: Optional[list] = None
    gate_importance: Optional[list] = None
    expert_activations: Optional[torch.FloatTensor] = None  # 新增专家激活统计


class BaseMoELayer(nn.Module):
    def __init__(self):
        super(BaseMoELayer, self).__init__()
        self.register_buffer('expert_activation_counts', None)
        self.register_buffer('total_samples', None)

        self.gate: Union[
            UniformPlainGate,
            UniformLearnableGate,
            RandomPlainGate,
            RandomLearnableGate,
            TopKBalancedNoisyGate,
            SwitchBalancedGate,
        ]
        self.calculator: Union[
            UniformCalculator, UniversalCalculator, SwitchDropTokenCalculator
        ]

    def init_expert_stats(self):
        """Initialize expert statistics tracking"""
        if self.expert_activation_counts is None:
            device = next(self.parameters()).device  # 获取模型当前设备
            self.register_buffer('expert_activation_counts', torch.zeros(self.num_experts, device=device))
            self.register_buffer('total_samples', torch.tensor(0, device=device))

    def _create_gate(self, **kwargs):
        self.gate_type = kwargs.get("gate_type", "TopKBalancedNoisyGate")

        if self.gate_type == "UniformPlainGate":  # all select gate
            self.gate = UniformPlainGate(
                self.input_size,
                self.num_experts,
                use_softmax=kwargs.get("gate_use_softmax", False),
            )
        elif self.gate_type == "UniformLearnableGate":  # all select gate with network
            self.gate = UniformLearnableGate(
                self.input_size,
                self.num_experts,
                gate_network=kwargs.get("gate_network", "mlp"),
                use_softmax=kwargs.get("gate_use_softmax", False),
            )
        elif self.gate_type == "RandomPlainGate":  # random select gate
            self.gate = RandomPlainGate(
                self.input_size,
                self.num_experts,
                self.num_selects,
                use_softmax=kwargs.get("gate_use_softmax", False),
            )
        elif self.gate_type == "RandomLearnableGate":  # random gate with network
            self.gate = RandomLearnableGate(
                self.input_size,
                self.num_experts,
                self.num_selects,
                gate_network=kwargs.get("gate_network", "mlp"),
                use_softmax=kwargs.get("gate_use_softmax", False),
                add_noise=kwargs.get("gate_add_noise", True),
                noise_epsilon=kwargs.get("gate_noise_epsilon", 1e-2),
            )
        elif self.gate_type == "TopKBalancedNoisyGate":  # noisy gate
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
        elif self.gate_type == "SwitchBalancedGate":  # switch gate
            self.gate = SwitchBalancedGate(
                self.input_size,
                self.num_experts,
                self.num_selects,
                gate_network=kwargs.get("gate_network", "mlp"),
                use_softmax=kwargs.get("gate_use_softmax", True),
                use_balance=kwargs.get("gate_use_balance", True),
                balance_loss_weight=kwargs.get("gate_balance_loss_weight", 1e-2),
                add_noise=kwargs.get("gate_add_noise", True),
            )
        else:
            raise NotImplementedError

    def _create_calculator(self, experts, **kwargs):
        self.calculator_type = kwargs.get("calculator_type", "UniversalCalculator")

        if self.calculator_type == "UniformCalculator":  # all select calculator
            self.calculator = UniformCalculator(
                experts,
                multiply_gate_scores=kwargs.get("multiply_gate_scores", True),
                score_scale_factor=kwargs.get("score_scale_factor", 1.0),
            )
        elif self.calculator_type == "UniversalCalculator":  # top K calculator
            self.calculator = UniversalCalculator(
                experts,
                multiply_gate_scores=kwargs.get("multiply_gate_scores", True),
                score_scale_factor=kwargs.get("score_scale_factor", 1.0),
                add_weight_norm=kwargs.get("add_weight_norm", False),
            )
        elif self.calculator_type == "SwitchDropTokenCalculator":  # switch calculator
            self.calculator = SwitchDropTokenCalculator(
                experts,
                multiply_gate_scores=kwargs.get("multiply_gate_scores", True),
                score_scale_factor=kwargs.get("score_scale_factor", 1.0),
                drop_tokens=kwargs.get("drop_tokens", True),
                dropped_padding=kwargs.get("dropped_padding", "zero"),
                capacity_factor=kwargs.get("capacity_factor", 1.25),
                add_weight_norm=kwargs.get("add_weight_norm", False),
            )
        else:
            raise NotImplementedError

    def forward(self, x) -> MoEMlpOutput:
        self.init_expert_stats()
        original_shape = x.shape[:-1]
        x = x.reshape(-1, self.input_size)
        
        # 原始gate计算
        gate_outputs: dict = self.gate(x)
        
        # 记录专家激活情况
        if 'indices' in gate_outputs:  # TopK类gate
            expert_indices = gate_outputs['topK_indices']  # [batch*seq_len, num_selects]
            # 使用bincount统计每个专家被选择的次数
            counts = torch.bincount(expert_indices.flatten(), minlength=self.num_experts)
            self.expert_activation_counts += counts.float()
        elif 'probs' in gate_outputs:  # 概率类gate
            expert_probs = gate_outputs['probs']  # [batch*seq_len, num_experts]
            self.expert_activation_counts += expert_probs.sum(dim=0)
        
        self.total_samples += x.size(0)
        
        # 原始计算逻辑
        calc_outs: CalculatorOutput = self.calculator(x, **gate_outputs)
        y = calc_outs.hidden_states.reshape(original_shape + (self.output_size,))
        
        # 计算当前批次的专家激活率
        current_activations = None
        if 'indices' in gate_outputs:
            current_activations = torch.bincount(expert_indices.flatten(), minlength=self.num_experts).float() / x.size(0)
        elif 'probs' in gate_outputs:
            current_activations = expert_probs.mean(dim=0)
        
        return MoEMlpOutput(
            hidden_states=y,
            balance_loss=gate_outputs.get("balance_loss"),
            num_dropped_tokens=calc_outs.num_dropped_tokens,
            gate_load=gate_outputs.get("load"),
            gate_importance=gate_outputs.get("importance"),
            expert_activations=current_activations  # 返回当前批次的激活率
        )
    
    def reset_expert_stats(self):
        """重置统计计数器"""
        if self.expert_activation_counts is not None:
            self.expert_activation_counts.zero_()
            self.total_samples.zero_()

    def get_expert_activations(self):
        """获取专家激活统计"""
        if self.total_samples > 0:
            return self.expert_activation_counts / self.total_samples
        return torch.zeros_like(self.expert_activation_counts)

    # fmt: off
    def set_num_selects(self, num_selects):
        if "num_selects" not in vars(self.gate):
            raise KeyError(f'{self.gate_type} does not have a key named "num_selects".')
        elif num_selects > self.gate.num_experts:
            raise ValueError('The value of "num_selects" must satisfy "num_selects <= num_experts"!')
        elif self.gate_type in ("SwitchBalancedGate",):
            raise ValueError(f"{self.gate_type} doesn't support manually setting num_selects.")
        else:
            self.num_selects = num_selects
            self.gate.num_selects = num_selects

    def set_gate_use_softmax(self, use_softmax):
        if "use_softmax" not in vars(self.gate):
            raise KeyError(f'{self.gate_type} does not have a key named "use_softmax".')
        else:
            self.gate.use_softmax = use_softmax

    def set_gate_use_balance(self, use_balance):
        if "use_balance" not in vars(self.gate):
            raise KeyError(f'{self.gate_type} does not have a key named "use_balance".')
        else:
            self.gate.use_balance = use_balance

    def set_gate_balance_loss_weight(self, balance_loss_weight):
        if "balance_loss_weight" not in vars(self.gate):
            raise KeyError(f'{self.gate_type} does not have a key named "balance_loss_weight".')
        else:
            self.gate.balance_loss_weight = balance_loss_weight

    def set_gate_add_noise(self, add_noise):
        if "add_noise" not in vars(self.gate):
            raise KeyError(f'{self.gate_type} does not have a key named "add_noise".')
        else:
            self.gate.add_noise = add_noise

    def set_gate_noise_epsilon(self, noise_epsilon):
        if "noise_epsilon" not in vars(self.gate):
            raise KeyError(f'{self.gate_type} does not have a key named "noise_epsilon".')
        else:
            self.gate.noise_epsilon = noise_epsilon

    def set_calculator_multiply_gate_scores(self, multiply_gate_scores):
        if "multiply_gate_scores" not in vars(self.calculator):
            raise KeyError(f'{self.gate_type} does not have a key named "multiply_gate_scores".')
        else:
            self.calculator.multiply_gate_scores = multiply_gate_scores

    def set_calculator_score_scale_factor(self, score_scale_factor):
        if "score_scale_factor" not in vars(self.calculator):
            raise KeyError(f'{self.gate_type} does not have a key named "score_scale_factor".')
        else:
            self.calculator.score_scale_factor = score_scale_factor

    def set_calculator_drop_tokens(self, drop_tokens):
        if "drop_tokens" not in vars(self.calculator):
            raise KeyError(f'{self.gate_type} does not have a key named "drop_tokens".')
        elif drop_tokens and self.calculator.dropped_padding != "zero" and self.input_size != self.output_size:
            warnings.warn('Setting "drop_tokens=True" without zero dropped padding when "input_size != output_size" will cause error!')
        else:
            self.calculator.drop_tokens = drop_tokens

    def set_calculator_dropped_padding(self, dropped_padding):
        if "dropped_padding" not in vars(self.calculator):
            raise KeyError(f'{self.gate_type} does not have a key named "dropped_padding".')
        elif dropped_padding not in self.calculator.available_dropped_padding_choices:
            raise ValueError(f"'dropped_padding' type not available! (available choices: {self.calculator.available_dropped_padding_choices})")
        elif self.calculator.drop_tokens and dropped_padding != "zero" and self.input_size != self.output_size:
            warnings.warn(f'Setting "dropped_padding={dropped_padding}" with "drop_tokens=True" when "input_size != output_size" will cause error!')
        else:
            self.calculator.dropped_padding = dropped_padding

    def set_calculator_capacity_factor(self, capacity_factor):
        if "capacity_factor" not in vars(self.calculator):
            raise KeyError(f'{self.gate_type} does not have a key named "capacity_factor".')
        else:
            self.calculator.capacity_factor = capacity_factor

    def reset_gate_network(self):
        self.gate.reset_gate_network()

    def reset_experts(self):
        self.calculator.reset_experts()
    # fmt: on


class LinearMoELayer(BaseMoELayer):
    def __init__(
        self, input_size, output_size, num_experts, num_selects, bias=True, **kwargs
    ):
        # fmt: off
        super(LinearMoELayer, self).__init__()
        assert (num_selects <= num_experts)  # 选择数量大于专家数量，报错
        self.input_size = input_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.num_selects = num_selects
        self.bias = bias

        experts = LinearExperts(
            input_size,
            output_size,
            num_experts,
            bias=bias,
        )

        self._create_gate(**kwargs)
        self._create_calculator(experts, **kwargs)
        # fmt: on


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

        self._create_gate(**kwargs)
        self._create_calculator(experts, **kwargs)
        # fmt: on