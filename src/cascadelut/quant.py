#  This file is part of CascadeLUT.
#
#  CascadeLUT is a derivative work based on LogicNets,
#  which is licensed under the Apache License 2.0.

#  Copyright (C) 2021 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import math
import itertools
from functools import reduce
from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from brevitas.core.quant import QuantType
from brevitas.core.quant import RescalingIntQuant, ClampedBinaryQuant

from .verilog import cascade_gen_layer_verilog, cascade_gen_neuron_verilog
from .bench import cascade_gen_neuron_bench


# TODO: Put this inside an abstract base class
def get_int_state_space(bits: int, signed: bool, narrow_range: bool, is_cuda: bool):
    start = int(
        0 if not signed else (-(2 ** (bits - 1)) + int(narrow_range))
    )  # calculate the minimum value in the range
    end = int(
        start + 2 ** (bits) - int(narrow_range)
    )  # calculate the maximum of the range
    state_space = torch.as_tensor(range(start, end))
    if is_cuda:
        return state_space.cuda()
    return state_space


# TODO: Put this inside an abstract base class
def get_float_state_space(
    bits: int,
    scale_factor: float,
    signed: bool,
    narrow_range: bool,
    quant_type: QuantType,
    is_cuda: bool,
):
    if quant_type == QuantType.INT:
        bin_state_space = get_int_state_space(bits, signed, narrow_range, is_cuda)
    elif quant_type == QuantType.BINARY:
        bin_state_space = torch.as_tensor([-1.0, 1.0])
    if is_cuda:
        bin_state_space = bin_state_space.cuda()
    state_space = scale_factor * bin_state_space
    return state_space

# TODO: Add an abstract class with a specific interface which all brevitas-based classes inherit from?
class QuantBrevitasActivation(nn.Module):
    def __init__(
        self, brevitas_module, pre_transforms: list = [], post_transforms: list = [], cuda: bool = True
    ):
        super(QuantBrevitasActivation, self).__init__()
        self.brevitas_module = brevitas_module
        self.pre_transforms = nn.ModuleList(pre_transforms)
        self.post_transforms = nn.ModuleList(post_transforms)
        self.is_bin_output = False
        self.cuda = cuda

    # TODO: Move to a base class
    # TODO: Move the string templates to verilog.py

    def get_bin_str_from_float(self, x, is_cuda):
        quant_type = self.get_quant_type()
        _, bits = self.get_scale_factor_bits(is_cuda)
        if quant_type == QuantType.INT:
            tensor_quant = (
                self.brevitas_module.act_quant.fused_activation_quant_proxy.tensor_quant
            )
            narrow_range = tensor_quant.int_quant.narrow_range
            signed = tensor_quant.int_quant.signed
            offset = 2 ** (bits - 1) - int(narrow_range) if signed else 0
            for idx, value in enumerate(self.get_state_space(is_cuda)):
                if math.isclose(self.get_state_space(is_cuda)[idx],x,rel_tol=1e-03):
                    return f"{int(self.get_bin_state_space(is_cuda)[idx]+offset):0{int(bits)}b}"
            raise Exception("Value not found in state space")
        elif quant_type == QuantType.BINARY:
            return f"{int(x):0{int(bits)}b}"
        else:
            raise Exception("Unknown quantization type: {}".format(quant_type))

    def get_bin_str_from_int(self, x, is_cuda):
        quant_type = self.get_quant_type()
        scale_factor, bits = self.get_scale_factor_bits(is_cuda)
        if quant_type == QuantType.INT:
            tensor_quant = (
                self.brevitas_module.act_quant.fused_activation_quant_proxy.tensor_quant
            )
            narrow_range = tensor_quant.int_quant.narrow_range
            signed = tensor_quant.int_quant.signed
            offset = 2 ** (bits - 1) - int(narrow_range) if signed else 0
            if int(x) - x != 0:
                raise Exception("Value is not an integer, either run lut_inference first or change function to get_bin_str_from_float")
            return f"{int(x+offset):0{int(bits)}b}"
        elif quant_type == QuantType.BINARY:
            return f"{int(x):0{int(bits)}b}"
        else:
            raise Exception("Unknown quantization type: {}".format(quant_type))

    # TODO: Move to a base class
    def bin_output(self):
        self.is_bin_output = True

    # TODO: Move to a base class
    def float_output(self):
        self.is_bin_output = False

    def get_quant_type(self):
        brevitas_module_type = type(
            self.brevitas_module.act_quant.fused_activation_quant_proxy.tensor_quant
        )
        if brevitas_module_type == RescalingIntQuant:
            return QuantType.INT
        elif brevitas_module_type == ClampedBinaryQuant:
            return QuantType.BINARY
        else:
            raise Exception(
                "Unknown quantization type for tensor_quant: {}".format(
                    brevitas_module_type
                )
            )

    # TODO: Allow for different bitwidths / scales per output
    def get_scale_factor_bits(self, is_cuda):
        # TODO: put guards in this based on quantization type
        quant_proxy = self.brevitas_module.act_quant
        current_status = quant_proxy.training
        quant_proxy.eval()
        zero_point = torch.Tensor([0.0])
        if is_cuda:
            zero_point = zero_point.cuda()
        _, scale_factor, _, bits, _, _ = quant_proxy(zero_point)
        quant_proxy.training = current_status
        return scale_factor, bits

    # Return a floating point version of the state space, this return values
    # that PyTorch would see at the output of this layer during training.
    # TODO: Merge this function with 'get_bin_state_space' and remove duplicated code.
    def get_state_space(self, is_cuda):
        quant_type = self.get_quant_type()
        scale_factor, bits = self.get_scale_factor_bits(is_cuda)
        if quant_type == QuantType.INT:
            tensor_quant = (
                self.brevitas_module.act_quant.fused_activation_quant_proxy.tensor_quant
            )
            narrow_range = tensor_quant.int_quant.narrow_range
            signed = tensor_quant.int_quant.signed
            state_space = get_float_state_space(
                bits, scale_factor, signed, narrow_range, quant_type, is_cuda
            )
        elif quant_type == QuantType.BINARY:
            state_space = scale_factor * torch.tensor([-1, 1])
        else:
            raise Exception("Unknown quantization type: {}".format(quant_type))
        return self.apply_post_transforms(state_space)

    # Return the underlying binary representation of the values returned by
    # 'get_state_space'
    def get_bin_state_space(self, is_cuda):
        quant_type = self.get_quant_type()
        _, bits = self.get_scale_factor_bits(is_cuda)
        if quant_type == QuantType.INT:
            tensor_quant = (
                self.brevitas_module.act_quant.fused_activation_quant_proxy.tensor_quant
            )
            narrow_range = tensor_quant.int_quant.narrow_range
            signed = tensor_quant.int_quant.signed
            state_space = get_int_state_space(bits, signed, narrow_range, is_cuda)
        elif quant_type == QuantType.BINARY:
            state_space = torch.tensor([0, 1])
        else:
            raise Exception("Unknown quantization type: {}".format(quant_type))
        return state_space

    def apply_pre_transforms(self, x):
        for i in range(len(self.pre_transforms)):
            x = self.pre_transforms[i](x)
        return x

    def apply_post_transforms(self, x):
        for i in range(len(self.post_transforms)):
            x = self.post_transforms[i](x)
        return x

    def forward(self, x):
        if self.is_bin_output:
            s, _ = self.get_scale_factor_bits(self.cuda)
            x = self.apply_pre_transforms(x)
            x = self.brevitas_module(x)
            x = torch.round(x / s).type(torch.int64)
        else:
            x = self.apply_pre_transforms(x)
            x = self.brevitas_module(x)
            x = self.apply_post_transforms(x)
        return x
    

def _generate_permutation_matrix_multi(state_spaces: list, cuda: bool) -> Tensor:
    combos = list(itertools.product(*[s.tolist() for s in state_spaces]))
    mat = torch.tensor(combos, dtype=torch.float32)
    if cuda:
        mat = mat.cuda()
    return mat


def _generate_bin_permutation_matrix_multi(bin_state_spaces: list, cuda: bool) -> Tensor:
    combos = list(itertools.product(*[s.tolist() for s in bin_state_spaces]))
    mat = torch.tensor(combos, dtype=torch.int64)
    if cuda:
        mat = mat.cuda()
    return mat


def generate_neuron_connection_verilog_composite(input_indices, composite_quant):
    connection_string = ""
    num_indices = len(input_indices)
    for i in range(num_indices):
        index = int(input_indices[i])
        offset = composite_quant.get_bit_offset_for_feature(index)
        bits = composite_quant.get_bits_for_feature(index)
        for b in reversed(range(bits)):
            connection_string += f"M0[{offset + b}]"
            if not (i == num_indices - 1 and b == 0):
                connection_string += ", "
    return connection_string


class LearnablePerFeatureMultiBitActivation(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_bits: int = 1,
        initial_scale: float = 1.0,      # kept for backward compat; ignored
        initial_threshold: float = 0.0,
        initial_gap: float = 1.0,
        initial_log_temp: float = 0.0,   # kept for compat; temperature preferred
        temperature: float = 1.0,
        initial_bias: float = 0.0,
        cuda: bool = False,
    ):
        super().__init__()
        if num_bits < 1:
            raise ValueError(f"num_bits must be >= 1, got {num_bits}")

        self.num_features = num_features
        self.num_bits = num_bits
        self.num_levels = 2 ** num_bits
        self.num_thresholds = self.num_levels - 1   # 2^k − 1
        self._cuda = cuda
        self.is_bin_output = False

        # scale removed: hard code depends only on sign of (x - threshold)
        t0 = float(initial_threshold) - float(initial_bias)   # scale=1 implicit
        self.base_threshold = nn.Parameter(torch.full((num_features,), t0))

        if self.num_thresholds > 1:
            raw_gap_init = math.log(max(math.expm1(float(initial_gap)), 1e-9))
            self.raw_gaps = nn.Parameter(
                torch.full((num_features, self.num_thresholds - 1), raw_gap_init)
            )
        else:
            self.register_parameter("raw_gaps", None)

        init_temp = float(temperature) if temperature != 1.0 else math.exp(float(initial_log_temp))
        self.register_buffer("_temp", torch.tensor(max(init_temp, 1e-4)))

    def _sorted_thresholds(self) -> Tensor:
        t0 = self.base_threshold.unsqueeze(-1)          # (F, 1)
        if self.raw_gaps is None:
            # 1-bit: single threshold per feature
            return t0                                   # (F, 1)

        gaps = F.softplus(self.raw_gaps)                # (F, T-1)  strictly positive
        cum_gaps = torch.cumsum(gaps, dim=-1)           # (F, T-1)
        # thresholds: [t0, t0+g0, t0+g0+g1, …]
        return torch.cat([t0, t0 + cum_gaps], dim=-1)  # (F, T)

    @property
    def thresholds(self) -> Tensor:
        t = self._sorted_thresholds().detach()
        return t.squeeze(-1) if self.num_bits == 1 else t

    @thresholds.setter
    def thresholds(self, value):
        with torch.no_grad():
            v = torch.as_tensor(value, dtype=self.base_threshold.dtype)
            if v.dim() > 1:
                v = v[:, 0]
            self.base_threshold.copy_(v.squeeze())

    @property
    def threshold_bias(self) -> Tensor:
        return -self.base_threshold.detach()

    @threshold_bias.setter
    def threshold_bias(self, value):
        with torch.no_grad():
            v = torch.as_tensor(value, dtype=self.base_threshold.dtype)
            self.base_threshold.copy_(-v.squeeze())

    @property
    def temperature(self) -> float:
        return float(self._temp.item())

    @temperature.setter
    def temperature(self, value: float):
        self._temp.fill_(max(float(value), 1e-4))

    def forward(self, x: Tensor) -> Tensor:
        thresholds = self._sorted_thresholds().to(x.device)  # (F, T)
        diff = x.unsqueeze(-1) - thresholds                  # (..., F, T)

        hard_bits = (diff >= 0.0).to(x.dtype)
        hard_code = hard_bits.sum(-1)                         # (..., F) ∈ {0,...,K-1}

        if self.is_bin_output:
            return hard_code.long()

        denom = max(self.num_levels - 1, 1)
        hard_level = -1.0 + 2.0 * hard_code / denom          # ∈ {-1,...,+1}

        if not self.training:
            return hard_level

        tau = self._temp.to(x.device).clamp_min(0.05)
        soft_bits = torch.sigmoid(diff / tau)                 # (..., F, T)
        soft_code = soft_bits.sum(-1)
        soft_level = -1.0 + 2.0 * soft_code / denom

        return soft_level + (hard_level - soft_level).detach()

    def bin_output(self):
        self.is_bin_output = True

    def float_output(self):
        self.is_bin_output = False

    def get_state_space(self, is_cuda: bool) -> Tensor:
        ss = torch.linspace(-1.0, 1.0, self.num_levels)
        return ss.cuda() if is_cuda else ss

    def get_bin_state_space(self, is_cuda: bool) -> Tensor:
        bs = torch.arange(self.num_levels, dtype=torch.int64)
        return bs.cuda() if is_cuda else bs

    def get_scale_factor_bits(self, is_cuda: bool):
        s = torch.tensor(1.0)
        b = torch.tensor(float(self.num_bits))
        if is_cuda:
            s, b = s.cuda(), b.cuda()
        return s, b

    def get_bin_str_from_int(self, x, is_cuda: bool) -> str:
        v = int(x)
        if not (0 <= v < self.num_levels):
            raise ValueError(f"Expected 0..{self.num_levels - 1}, got {x}")
        return f"{v:0{self.num_bits}b}"

    def get_bin_str_from_float(self, x, is_cuda: bool) -> str:
        # x is a latent level in linspace(-1, +1, num_levels); convert to code
        denom = max(self.num_levels - 1, 1)
        v = int(round((float(x) + 1.0) / 2.0 * denom))
        if not (0 <= v < self.num_levels):
            raise ValueError(f"Expected level in [-1.0, 1.0], got {x}")
        return f"{v:0{self.num_bits}b}"

    def apply_pre_transforms(self, x):
        return x

    def apply_post_transforms(self, x):
        return x

class CascadeCompositeInputQuant(nn.Module):
    """Wraps an optional hidden quantizer + one or more chunk quantizers for cascade input."""
    def __init__(
        self,
        hidden_quant,
        hidden_size: int,
        chunk_quants: list,
        chunk_sizes: list,
        cuda: bool = False,
    ):
        super().__init__()
        assert len(chunk_quants) == len(chunk_sizes)
        self.hidden_quant = hidden_quant
        self.hidden_size = hidden_size
        self.chunk_quants = nn.ModuleList(chunk_quants)
        self.chunk_sizes = chunk_sizes
        self.total_size = hidden_size + sum(chunk_sizes)
        self._cuda = cuda
        self.is_bin_output = False

        self._slices: List = []
        offset = 0
        if hidden_size > 0 and hidden_quant is not None:
            self._slices.append((hidden_quant, offset, offset + hidden_size))
            offset += hidden_size
        for q, sz in zip(chunk_quants, chunk_sizes):
            self._slices.append((q, offset, offset + sz))
            offset += sz

    def _quant_for_feature(self, idx: int):
        for quant, start, end in self._slices:
            if start <= idx < end:
                return quant
        raise ValueError(f"Feature index {idx} out of range [0, {self.total_size})")

    def _is_chunk_idx(self, idx: int) -> bool:
        return idx >= self.hidden_size

    def forward(self, x):
        parts = []
        for quant, start, end in self._slices:
            parts.append(quant(x[:, start:end]))
        return torch.cat(parts, dim=1)

    def bin_output(self):
        self.is_bin_output = True
        if self.hidden_quant is not None:
            self.hidden_quant.bin_output()
        for q in self.chunk_quants:
            q.bin_output()

    def float_output(self):
        self.is_bin_output = False
        if self.hidden_quant is not None:
            self.hidden_quant.float_output()
        for q in self.chunk_quants:
            q.float_output()

    def get_state_space(self, is_cuda):
        ref = self.hidden_quant if self.hidden_quant is not None else self.chunk_quants[0]
        return ref.get_state_space(is_cuda)

    def get_bin_state_space(self, is_cuda):
        ref = self.hidden_quant if self.hidden_quant is not None else self.chunk_quants[0]
        return ref.get_bin_state_space(is_cuda)

    def get_scale_factor_bits(self, is_cuda):
        ref = self.hidden_quant if self.hidden_quant is not None else self.chunk_quants[0]
        return ref.get_scale_factor_bits(is_cuda)

    def get_bin_str_from_int(self, x, is_cuda):
        ref = self.hidden_quant if self.hidden_quant is not None else self.chunk_quants[0]
        return ref.get_bin_str_from_int(x, is_cuda)

    def get_bin_str_from_float(self, x, is_cuda):
        ref = self.hidden_quant if self.hidden_quant is not None else self.chunk_quants[0]
        return ref.get_bin_str_from_float(x, is_cuda)

    def apply_pre_transforms(self, x):
        return x

    def apply_post_transforms(self, x):
        return x

    def get_per_slot_state_spaces(self, fan_in, imask, is_cuda):
        return [
            self._quant_for_feature(int(imask[j])).get_state_space(is_cuda)
            for j in range(fan_in)
        ]

    def get_per_slot_bin_state_spaces(self, fan_in, imask, is_cuda):
        return [
            self._quant_for_feature(int(imask[j])).get_bin_state_space(is_cuda)
            for j in range(fan_in)
        ]

    def get_bin_str_for_slot(self, global_idx: int, val, is_cuda: bool) -> str:
        return self._quant_for_feature(global_idx).get_bin_str_from_int(val, is_cuda)

    def get_bits_for_feature(self, feature_idx: int) -> int:
        _, bits = self._quant_for_feature(feature_idx).get_scale_factor_bits(self._cuda)
        return int(bits)

    def get_bit_offset_for_feature(self, feature_idx: int) -> int:
        bit_offset = 0
        for quant, start, end in self._slices:
            _, bits = quant.get_scale_factor_bits(self._cuda)
            bits = int(bits)
            if feature_idx < end:
                return bit_offset + (feature_idx - start) * bits
            bit_offset += (end - start) * bits
        raise ValueError(f"Feature index {feature_idx} out of range")

    def get_total_input_bits(self) -> int:
        total = 0
        for quant, start, end in self._slices:
            _, bits = quant.get_scale_factor_bits(self._cuda)
            total += (end - start) * int(bits)
        return total



def cascade_calculate_truth_tables(self):
    if self.input_quant is None:
        self.neuron_truth_tables = None
        return

    composite = self.input_quant

    with torch.no_grad():
        apply_iq, apply_oq = self.apply_input_quant, self.apply_output_quant
        self.apply_input_quant = False
        self.apply_output_quant = False
        is_bin_output = self.output_quant.is_bin_output
        self.output_quant.float_output()

        h_ss = composite.hidden_quant.get_state_space(self.cuda)
        c_ss = composite.chunk_quants[0].get_state_space(self.cuda)
        h_bs = composite.hidden_quant.get_bin_state_space(self.cuda)
        c_bs = composite.chunk_quants[0].get_bin_state_space(self.cuda)

        # group neurons by which fan_in slots are chunk vs hidden — avoids recomputing
        # permutation matrices for neurons with the same input-type pattern
        patterns = []
        for n in range(self.out_features):
            neuron_imask = self.imask[n]
            pattern = tuple(
                1 if composite._is_chunk_idx(int(neuron_imask[j])) else 0
                for j in range(self.fan_in)
            )
            patterns.append(pattern)

        pattern_groups = defaultdict(list)
        for n, p in enumerate(patterns):
            pattern_groups[p].append(n)

        neuron_truth_tables = [None] * self.out_features

        for pattern, group_neurons in pattern_groups.items():
            float_ss_list = [c_ss if p == 1 else h_ss for p in pattern]
            bin_ss_list   = [c_bs if p == 1 else h_bs for p in pattern]

            float_perm = _generate_permutation_matrix_multi(float_ss_list, self.cuda)
            bin_perm   = _generate_bin_permutation_matrix_multi(bin_ss_list, self.cuda)

            if self.cuda:
                float_perm = float_perm.cuda()
                bin_perm   = bin_perm.cuda()

            float_out = self.output_quant(self.forward_to_fill_luts(float_perm))
            self.output_quant.bin_output()
            bin_out = self.output_quant(self.forward_to_fill_luts(float_perm))
            self.output_quant.float_output()

            for n in group_neurons:
                neuron_truth_tables[n] = (
                    self.imask[n],
                    bin_perm,
                    float_out[:, n],
                    bin_out[:, n],
                )

        self.output_quant.is_bin_output = is_bin_output
        self.apply_input_quant = apply_iq
        self.apply_output_quant = apply_oq

    self.neuron_truth_tables = neuron_truth_tables

def patch_cascade_layer(layer) -> None:
    import types
    if not isinstance(layer.input_quant, CascadeCompositeInputQuant):
        return
    layer.calculate_truth_tables = types.MethodType(cascade_calculate_truth_tables, layer)
    layer.gen_layer_verilog       = types.MethodType(cascade_gen_layer_verilog,       layer)
    layer.gen_neuron_verilog      = types.MethodType(cascade_gen_neuron_verilog,      layer)
    layer.gen_neuron_bench        = types.MethodType(cascade_gen_neuron_bench,        layer)


class ThresholdTemperatureScheduler:
    """
    Phase 1 (epochs 0 .. flat_start_epochs-1): constant at start_temp.
    Phase 2 (flat_start_epochs .. flat_start_epochs+anneal_epochs-1): cosine
             from start_temp down to end_temp.
    Phase 3 (beyond): constant at end_temp.
    """
    def __init__(
        self,
        model: nn.Module,
        start_temp: float = 1.0,
        end_temp: float = 0.10,
        anneal_epochs: int = 400,
        flat_start_epochs: int = 0,
        override: bool = True,
    ):
        self.modules = [
            m for m in model.modules()
            if isinstance(m, LearnablePerFeatureMultiBitActivation)
        ]
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.anneal_epochs = anneal_epochs
        self.flat_start_epochs = flat_start_epochs
        self.override = override

    def _scheduled_temp(self, epoch: int) -> float:
        if epoch < self.flat_start_epochs:
            return self.start_temp
        adjusted = epoch - self.flat_start_epochs
        progress = min(adjusted / max(self.anneal_epochs, 1), 1.0)
        t = self.end_temp + (self.start_temp - self.end_temp) * \
            0.5 * (1.0 + math.cos(math.pi * progress))
        return max(t, self.end_temp)

    def step(self, epoch: int):
        temp = self._scheduled_temp(epoch)
        for m in self.modules:
            if self.override:
                m.temperature = temp
            else:
                m.temperature = min(m.temperature, temp)

    def step_with_lr(self, lr: float, base_lr: float) -> None:
        if base_lr <= 0:
            return
        ratio = max(0.0, min(lr / base_lr, 1.0))
        temp = self.end_temp + (self.start_temp - self.end_temp) * ratio
        for m in self.modules:
            if self.override:
                m.temperature = temp
            else:
                m.temperature = min(m.temperature, temp)

    def current_temperature(self) -> float:
        return self.modules[0].temperature if self.modules else 0.0