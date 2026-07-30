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

from functools import partial, reduce

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import itertools
import math, os
import numpy as np

from .init import random_restrict_fanin
from .util import fetch_mask_indices, generate_permutation_matrix
from .quant import CascadeCompositeInputQuant, patch_cascade_layer
from .verilog import (
    generate_lut_verilog,
    generate_neuron_connection_verilog,
    layer_connection_verilog,
    generate_logicnets_verilog,
    generate_register_verilog,
    generate_cascade_top_module,
)
from .bench import generate_lut_bench, generate_lut_input_string, sort_to_bench

# TODO: Create a container module which performs this function.
def generate_truth_tables(model: nn.Module, verbose: bool = False) -> None:
    training = model.training
    model.eval()

    # fix up input_quant for layers loaded from checkpoint with input_quant=None
    if hasattr(model, 'module_list'):
        for i, module in enumerate(model.module_list):
            if type(module) == SparseLinearNeq and module.input_quant is None:
                if i > 0 and hasattr(model.module_list[i-1], 'output_quant'):
                    module.input_quant = model.module_list[i-1].output_quant
                    if verbose:
                        print(f"Fixed input_quant for layer {i} from previous layer's output_quant")

    for name, module in model.named_modules():
        if type(module) == SparseLinearNeq:
            if isinstance(module.input_quant, CascadeCompositeInputQuant):
                patch_cascade_layer(module)
                if verbose:
                    print(f"Patched cascade layer {name}")

    for name, module in model.named_modules():
        if type(module) == SparseLinearNeq:
            if verbose:
                print(f"Calculating truth tables for {name}")
            module.calculate_truth_tables()
            if verbose:
                if module.neuron_truth_tables is not None:
                    print(
                        f"Truth tables generated for {len(module.neuron_truth_tables)} neurons"
                    )
                else:
                    print(f"Skipped truth table generation (no input_quant)")
    model.training = training


# TODO: Create a container module which performs this function.
def lut_inference(model: nn.Module) -> None:
    for name, module in model.named_modules():
        if type(module) == SparseLinearNeq:
            module.lut_inference()


# TODO: Create a container module which performs this function.
def neq_inference(model: nn.Module) -> None:
    for name, module in model.named_modules():
        if type(module) == SparseLinearNeq:
            module.neq_inference()


# TODO: Should this go in with the other verilog functions?
# TODO: Support non-linear topologies
def module_list_to_verilog_module(
    module_list: nn.ModuleList,
    module_name: str,
    output_directory: str,
    add_registers: bool = True,
    generate_bench: bool = False,
    cascade_config: dict = None,
    hw_bandwidth: int = None,   # port width in bits; None → chunk_bits (LOAD_CPC=1)
):
    input_bitwidth = None
    output_bitwidth = None
    module_contents = ""
    layer_output_bits = []
    
    for i in range(len(module_list)):
        m = module_list[i]
        if type(m) == SparseLinearNeq:
            module_prefix = f"layer{i}"
            module_input_bits, module_output_bits = m.gen_layer_verilog(
                module_prefix, output_directory, generate_bench=generate_bench
            )
            layer_output_bits.append(module_output_bits)
            if i == 0:
                input_bitwidth = module_input_bits
            elif i == len(module_list) - 1:
                output_bitwidth = module_output_bits
            
            if not (cascade_config and cascade_config.get('enabled', False)):
                module_contents += layer_connection_verilog(
                    module_prefix,
                    input_string=f"M{i}",
                    input_bits=module_input_bits,
                    output_string=f"M{i+1}",
                    output_bits=module_output_bits,
                    output_wire=i != len(module_list) - 1,
                    register=add_registers,
                )
        else:
            raise Exception(
                f"Expect type(module) == SparseLinearNeq, {type(m)} found"
            )
    reg_verilog = generate_register_verilog()
    with open(f"{output_directory}/myreg.v", "w") as f:
        f.write(reg_verilog)
    
    if cascade_config and cascade_config.get('enabled', False):
        features_per_cycle  = cascade_config.get('features_per_cycle', 112)
        cascade_chunks  = cascade_config.get('cascade_chunks')
        cascade_layers  = cascade_config.get('cascade_layers')

        _, input_bw = module_list[0].input_quant.get_scale_factor_bits(module_list[0].cuda)
        chunk_bits = features_per_cycle * int(input_bw)

        cascade_top = generate_cascade_top_module(
            top_name          = module_name,
            layer_output_bits = layer_output_bits,
            chunk_bits        = chunk_bits,
            cascade_chunks    = cascade_chunks,
            cascade_layers    = cascade_layers,
            hw_bandwidth      = hw_bandwidth,
        )
        with open(f"{output_directory}/{module_name}.v", "w") as f:
            f.write(cascade_top)
        print(f"Generated cascade top module: {output_directory}/{module_name}.v")
    else:
        top_logicnets = generate_logicnets_verilog(
            module_name=module_name,
            input_name="M0",
            input_bits=input_bitwidth,
            output_name=f"M{len(module_list)}",
            output_bits=output_bitwidth,
            module_contents=module_contents,
        )
        with open(f"{output_directory}/{module_name}.v", "w") as f:
            f.write(top_logicnets)


class SparseLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super(SparseLinear, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias
        )
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

    def forward(self, input: Tensor) -> Tensor:
        return (input * self.weight).sum(dim=-1) + self.bias

class DenseForward(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(DenseForward, self).__init__(in_features=in_features, out_features=out_features, bias=bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)
    
# TODO: Perhaps make this two classes, separating the LUT and NEQ code.
class SparseLinearNeq(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        input_quant,
        output_quant,
        imask,
        support,
        dense_forward,
        fan_in,
        width_n,
        apply_input_quant=True,
        apply_output_quant=True,
        cuda=False,
        features_per_cycle=None,
        importance_order=None,
        is_first_layer=False,
    ) -> None:
        super(SparseLinearNeq, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_quant = input_quant
        self.imask = imask
        self.fan_in = fan_in
        self.width_n = width_n
        self.support = support
        self.dense_forward = dense_forward
        self.features_per_cycle = features_per_cycle
        self.importance_order = importance_order
        self.is_first_layer = is_first_layer

        self.relu = nn.ReLU(inplace=True)
        if cuda:
            self.relu = self.relu.cuda()
        self.fc1 = SparseLinear(fan_in, out_features*width_n)
        self.fc4 = SparseLinear(width_n, out_features)
        self.res2 = SparseLinear(fan_in, out_features)
        self.fc1_dense = DenseForward(in_features, out_features*width_n)
        self.res2_dense = DenseForward(in_features, out_features)

        self.output_quant = output_quant
        self.is_lut_inference = False
        self.neuron_truth_tables = None
        self.apply_input_quant = apply_input_quant
        self.apply_output_quant = apply_output_quant
        self.cuda = cuda

    # TODO: Move the verilog string templates to elsewhere
    # TODO: Move this to another class
    # TODO: Update this code to support custom bitwidths per input/output
    def gen_layer_verilog(self, module_prefix, directory, generate_bench: bool = True):
        _, input_bitwidth = self.input_quant.get_scale_factor_bits(self.cuda)
        _, output_bitwidth = self.output_quant.get_scale_factor_bits(self.cuda)
        input_bitwidth, output_bitwidth = int(input_bitwidth), int(output_bitwidth)
        total_input_bits = self.in_features * input_bitwidth
        total_output_bits = self.out_features * output_bitwidth
        layer_contents = f"module {module_prefix} (input [{total_input_bits-1}:0] M0, output [{total_output_bits-1}:0] M1);\n\n"
        output_offset = 0
        for index in range(self.out_features):
            module_name = f"{module_prefix}_N{index}"
            indices, _, _, _ = self.neuron_truth_tables[index]
            neuron_verilog = self.gen_neuron_verilog(
                index, module_name
            )  # Generate the contents of the neuron verilog
            with open(f"{directory}/{module_name}.v", "w") as f:
                f.write(neuron_verilog)
            if generate_bench:
                neuron_bench = self.gen_neuron_bench(
                    index, module_name
                )  # Generate the contents of the neuron verilog
                with open(f"{directory}/{module_name}.bench", "w") as f:
                    f.write(neuron_bench)
            connection_string = generate_neuron_connection_verilog(
                indices, input_bitwidth
            )  # Generate the string which connects the synapses to this neuron
            wire_name = f"{module_name}_wire"
            connection_line = f"wire [{len(indices)*input_bitwidth-1}:0] {wire_name} = {{{connection_string}}};\n"
            inst_line = f"{module_name} {module_name}_inst (.M0({wire_name}), .M1(M1[{output_offset+output_bitwidth-1}:{output_offset}]));\n\n"
            layer_contents += connection_line + inst_line
            output_offset += output_bitwidth
        layer_contents += "endmodule"
        with open(f"{directory}/{module_prefix}.v", "w") as f:
            f.write(layer_contents)
        return total_input_bits, total_output_bits

    # TODO: Move the verilog string templates to elsewhere
    # TODO: Move this to another class
    def gen_neuron_verilog(self, index, module_name):
        (
            indices,
            input_perm_matrix,
            float_output_states,
            bin_output_states,
        ) = self.neuron_truth_tables[index]
        _, input_bitwidth = self.input_quant.get_scale_factor_bits(self.cuda)
        _, output_bitwidth = self.output_quant.get_scale_factor_bits(self.cuda)
        cat_input_bitwidth = len(indices) * input_bitwidth
        lut_string = ""
        num_entries = input_perm_matrix.shape[0]
        for i in range(num_entries):
            entry_str = ""
            for idx in range(len(indices)):
                val = input_perm_matrix[i, idx]
                entry_str += self.input_quant.get_bin_str_from_int(val, is_cuda=self.cuda)
            res_str = self.output_quant.get_bin_str_from_int(bin_output_states[i], is_cuda=self.cuda)
            lut_string += f"\t\t\t{int(cat_input_bitwidth)}'b{entry_str}: M1r = {int(output_bitwidth)}'b{res_str};\n"
        return generate_lut_verilog(
            module_name, int(cat_input_bitwidth), int(output_bitwidth), lut_string
        )

    # TODO: Move the string templates to bench.py
    # TODO: Move this to another class
    def gen_neuron_bench(self, index, module_name):
        (
            indices,
            input_perm_matrix,
            float_output_states,
            bin_output_states,
        ) = self.neuron_truth_tables[index]
        _, input_bitwidth = self.input_quant.get_scale_factor_bits(self.cuda)
        _, output_bitwidth = self.output_quant.get_scale_factor_bits(self.cuda)
        cat_input_bitwidth = len(indices) * input_bitwidth
        lut_string = ""
        num_entries = input_perm_matrix.shape[0]
        # Sort the input_perm_matrix to match the bench format
        input_state_space_bin_str = list(
            map(
                lambda y: list(map(lambda z: self.input_quant.get_bin_str_from_int(z, is_cuda=self.cuda), y)),
                input_perm_matrix,
            )
        )
        sorted_bin_output_states = sort_to_bench(
            input_state_space_bin_str, bin_output_states
        )
        # Generate the LUT for each output
        for i in range(int(output_bitwidth)):
            lut_string += f"M1[{i}]       = LUT 0x"
            output_bin_str = reduce(
                lambda b, c: b + c,
                map(
                    lambda a: self.output_quant.get_bin_str_from_int(a, is_cuda=self.cuda)[
                        int(output_bitwidth) - 1 - i
                    ],
                    sorted_bin_output_states,
                ),
            )
            lut_hex_string = f"{int(output_bin_str,2):0{int(num_entries/4)}x} "
            lut_string += lut_hex_string
            lut_string += generate_lut_input_string(int(cat_input_bitwidth))
        return generate_lut_bench(
            int(cat_input_bitwidth), int(output_bitwidth), lut_string
        )

    def lut_inference(self):
        self.is_lut_inference = True
        self.input_quant.bin_output()
        self.output_quant.bin_output()

    def neq_inference(self):
        self.is_lut_inference = False
        self.input_quant.float_output()
        self.output_quant.float_output()

    # TODO: This function might be a useful utility outside of this class..
    def table_lookup(
        self,
        connected_input: Tensor,
        input_perm_matrix: Tensor,
        bin_output_states: Tensor,
    ) -> Tensor:
        # effective fan_in from perm matrix — may be < self.fan_in for partially-connected neurons
        fan_in_size = input_perm_matrix.shape[1]
        ci_bcast = connected_input.unsqueeze(2)  # Reshape to B x Fan-in x 1
        pm_bcast = (
            input_perm_matrix.t().unsqueeze(0)
        )  # Reshape to 1 x Fan-in x InputStates
        pm_bcast = pm_bcast.to(ci_bcast.device)
        eq = (ci_bcast == pm_bcast).sum(
            dim=1
        ) == fan_in_size  # Create a boolean matrix which matches input vectors to possible input states
        matches = eq.sum(dim=1)  # Count the number of perfect matches per input vector
        if not (matches == torch.ones_like(matches, dtype=matches.dtype)).all():
            raise Exception(
                f"One or more vectors in the input is not in the possible input state space"
            )
        indices = torch.argmax(eq.type(torch.int64), dim=1)
        return bin_output_states[indices]

    def lut_forward(self, x: Tensor) -> Tensor:
        if self.cuda:
            # move to the layer's actual device, not necessarily cuda:0
            _p = next(self.parameters(), None)
            if _p is not None:
                x = x.to(_p.device)
        if self.apply_input_quant:
            x = self.input_quant(x)
        y = torch.zeros((x.shape[0], self.out_features))
        if self.cuda:
            y = y.to(x.device)
        for i in range(self.out_features):
            (
                indices,
                input_perm_matrix,
                float_output_states,
                bin_output_states,
            ) = self.neuron_truth_tables[i]
            indices = indices.to(x.device)
            # indices are only valid (non -1) global input positions
            connected_input = x[:, indices]
            y[:, i] = self.table_lookup(
                connected_input, input_perm_matrix, bin_output_states
            )
        return y

    def forward(self, x: Tensor) -> Tensor:
        if self.is_lut_inference:
            x = self.lut_forward(x)
        elif self.dense_forward:
            if self.apply_input_quant:
                x = self.input_quant(x)
            residual1 = self.res2_dense(x)
            x = self.fc1_dense(x)
            x = self.relu(x)
            x = x.reshape(x.size(0), int(x.size(1)/self.width_n), self.width_n)
            x = self.fc4(x)
            x = x + residual1
            if self.apply_output_quant:
                x = self.output_quant(x)
        else:
            if self.apply_input_quant:
                x = self.input_quant(x)
            batch_size = x.size(0)
            imask = self.imask
            if imask.device != x.device:
                imask = imask.to(x.device)
            flat_idx = torch.clamp(imask.reshape(-1), 0, x.size(1) - 1)
            x_selected = x[:, flat_idx].reshape(batch_size, self.out_features, self.fan_in)

            residual1 = self.res2(x_selected)
            x = x_selected.repeat(1,1,self.width_n).reshape(batch_size, self.out_features*self.width_n, self.fan_in)
            x = self.fc1(x)
            x = self.relu(x)
            x = x.reshape(batch_size, int(x.size(1)/self.width_n), self.width_n)
            x = self.fc4(x)
            x = x + residual1
            if self.apply_output_quant:
                x = self.output_quant(x)
        return x

    def forward_incremental(self, x: Tensor, cycle: int = 0) -> Tensor:
        # only inputs available at `cycle` are unmasked.
        if not self.is_first_layer:
            return self.forward(x)
        if self.features_per_cycle is None or self.importance_order is None:
            return self.forward(x)
        bits_available = min((cycle + 1) * self.features_per_cycle, self.in_features)
        available_indices = self.importance_order[:bits_available]
        masked_input = torch.zeros_like(x)
        if self.cuda:
            masked_input = masked_input.cuda()
            available_indices = available_indices.cuda()
        masked_input[:, available_indices] = x[:, available_indices]
        return self.forward(masked_input)

    def forward_to_fill_luts(self, x: Tensor) -> Tensor:
        if self.apply_input_quant:
            x = self.input_quant(x)
        
        if self.dense_forward:
            residual1 = self.res2_dense(x)
            x = self.fc1_dense(x)
            x = self.relu(x)
            x = x.reshape(x.size(0), int(x.size(1)/self.width_n), self.width_n)
            x = self.fc4(x)
            x = x + residual1
        else:
            x = x.repeat(1, self.out_features)
            x = x.reshape(x.shape[0], self.out_features, self.fan_in)
            residual1 = self.res2(x)
            x = x.repeat(1,1,self.width_n).reshape(x.size(0), x.size(1)*self.width_n, self.fan_in)
            x = self.fc1(x)
            x = self.relu(x)
            x = x.reshape(x.size(0), int(x.size(1)/self.width_n), self.width_n)
            x = self.fc4(x)
            x = x + residual1
        
        if self.apply_output_quant:
            x = self.output_quant(x)
        return x

    # Consider using masked_select instead of fetching the indices
    def calculate_truth_tables(self):
        # input_quant may be None for layers loaded from checkpoint before the model sets it
        if self.input_quant is None:
            self.neuron_truth_tables = None
            return
            
        with torch.no_grad():
            # Precalculate all of the input value permutations
            input_state_space = list()  # TODO: is a list the right data-structure here?
            bin_state_space = list()
            neuron_state_space = (
                self.input_quant.get_state_space(is_cuda=self.cuda)
            )  # TODO: this call should include the index of the element of interest
            bin_space = (
                self.input_quant.get_bin_state_space(is_cuda=self.cuda)
            )  # TODO: this call should include the index of the element of interest
            input_state_space.append(neuron_state_space)
            bin_state_space.append(bin_space)
            neuron_truth_tables = list()

            # Retrieve the possible state space of the current neuron
            connected_state_space = [input_state_space[0] for i in range(self.fan_in)]
            bin_connected_state_space = [bin_state_space[0] for i in range(self.fan_in)]
            # Generate a matrix containing all possible input states
            input_permutation_matrix = generate_permutation_matrix(
                connected_state_space, self.cuda
            )  # matrix of all input combinations
            if self.cuda:
                input_permutation_matrix = input_permutation_matrix.cuda()
            bin_input_permutation_matrix = generate_permutation_matrix(
                bin_connected_state_space, self.cuda
            )

            # TODO: Update this block to just run inference on the fc layer, once BN has been moved to output_quant
            apply_input_quant, apply_output_quant = (
                self.apply_input_quant,
                self.apply_output_quant,
            )
            self.apply_input_quant, self.apply_output_quant = False, False
            is_bin_output = self.output_quant.is_bin_output
            self.output_quant.float_output()
            step = input_permutation_matrix.shape[0]
            output_states = self.output_quant(
                self.forward_to_fill_luts(input_permutation_matrix[0:step, :])
            )
            for segment in range(step, input_permutation_matrix.shape[0], step):
                output_states = torch.cat(
                    (
                        output_states,
                        self.output_quant(
                            self.forward_to_fill_luts(
                                input_permutation_matrix[segment : segment + step, :]
                            )
                        ),
                    ),
                    0,
                )  # Calculate float for the current input
            self.output_quant.bin_output()
            bin_output_states = self.output_quant(
                self.forward_to_fill_luts(input_permutation_matrix[0:step, :])
            )  # Calculate bin for the current input
            for segment in range(step, input_permutation_matrix.shape[0], step):
                bin_output_states = torch.cat(
                    (
                        bin_output_states,
                        self.output_quant(
                            self.forward_to_fill_luts(
                                input_permutation_matrix[segment : segment + step, :]
                            )
                        ),
                    ),
                    0,
                )  # Calculate float for the current input
            self.output_quant.is_bin_output = is_bin_output
            self.apply_input_quant, self.apply_output_quant = (
                apply_input_quant,
                apply_output_quant,
            )
            
            if self.dense_forward:
                neuron_masks = []
                for n in range(self.out_features):
                    neuron_masks.append(torch.arange(self.fan_in))
            else:
                # group neurons by which imask slots are -1 (disconnected) so we only
                # enumerate valid input combos — avoids O(states^fan_in) for sparse neurons
                from collections import defaultdict
                _param = next(self.parameters(), None)
                _layer_device = _param.device if _param is not None else torch.device('cpu')

                pattern_groups = defaultdict(list)
                for n in range(self.out_features):
                    nim = self.imask[n]
                    pattern = tuple((int(nim[j]) < 0) for j in range(self.fan_in))
                    pattern_groups[pattern].append(n)

                neuron_truth_tables = [None] * self.out_features

                for pattern, group_ns in pattern_groups.items():
                    valid_slots = [j for j, is_invalid in enumerate(pattern) if not is_invalid]
                    eff_fan_in = len(valid_slots)

                    if eff_fan_in == 0:
                        # fully disconnected neuron — constant output
                        zero_input = torch.zeros(1, self.fan_in, dtype=torch.float32, device=_layer_device)
                        self.apply_input_quant, self.apply_output_quant = False, False
                        self.output_quant.float_output()
                        degen_float = self.output_quant(self.forward_to_fill_luts(zero_input))
                        self.output_quant.bin_output()
                        degen_bin = self.output_quant(self.forward_to_fill_luts(zero_input))
                        self.output_quant.is_bin_output = is_bin_output
                        self.apply_input_quant, self.apply_output_quant = (
                            apply_input_quant, apply_output_quant
                        )
                        zero_perm = torch.zeros(1, 0, dtype=degen_bin.dtype, device=_layer_device)
                        for n in group_ns:
                            neuron_truth_tables[n] = (
                                torch.zeros(0, dtype=torch.long),
                                zero_perm,
                                degen_float[0:1, n],
                                degen_bin[0:1, n],
                            )
                        continue

                    valid_float_ss = [input_state_space[0] for _ in range(eff_fan_in)]
                    valid_bin_ss   = [bin_state_space[0]   for _ in range(eff_fan_in)]

                    valid_float_perm = generate_permutation_matrix(valid_float_ss, self.cuda)
                    valid_bin_perm   = generate_permutation_matrix(valid_bin_ss,   self.cuda)
                    n_perms = valid_float_perm.shape[0]

                    # disconnected slots stay 0; valid slots filled below
                    full_float_perm = torch.zeros(
                        n_perms, self.fan_in,
                        dtype=valid_float_perm.dtype,
                        device=_layer_device,
                    )
                    valid_float_perm = valid_float_perm.to(_layer_device)
                    valid_bin_perm   = valid_bin_perm.to(_layer_device)
                    for local_j, global_j in enumerate(valid_slots):
                        full_float_perm[:, global_j] = valid_float_perm[:, local_j]

                    self.apply_input_quant, self.apply_output_quant = False, False
                    self.output_quant.float_output()
                    grp_float_out = self.output_quant(
                        self.forward_to_fill_luts(full_float_perm)
                    )
                    self.output_quant.bin_output()
                    grp_bin_out = self.output_quant(
                        self.forward_to_fill_luts(full_float_perm)
                    )
                    self.output_quant.is_bin_output = is_bin_output
                    self.apply_input_quant, self.apply_output_quant = (
                        apply_input_quant, apply_output_quant
                    )

                    for n in group_ns:
                        nim = self.imask[n]
                        valid_indices = nim[nim >= 0]
                        neuron_truth_tables[n] = (
                            valid_indices,
                            valid_bin_perm,
                            grp_float_out[:, n],
                            grp_bin_out[:, n],
                        )

                self.apply_input_quant, self.apply_output_quant = (
                    apply_input_quant,
                    apply_output_quant,
                )
                self.neuron_truth_tables = neuron_truth_tables
                return

            for n in range(self.out_features):
                # Append the connectivity, input permutations and output permutations to the neuron truth tables
                neuron_truth_tables.append(
                    (
                        neuron_masks[n],
                        bin_input_permutation_matrix,
                        output_states[:, n],
                        bin_output_states[:, n],
                    )
                )  # Change this to be the binary output states
        self.neuron_truth_tables = neuron_truth_tables



def SupportMask(out_features: int, fan_in: int):
    imask = torch.zeros(out_features, fan_in, dtype=torch.long)
    for i in range(out_features):
        start_idx = (i * fan_in) % (out_features * fan_in)
        for j in range(fan_in):
            imask[i, j] = (start_idx + j) % (out_features * fan_in)
    return imask


class ScalarScaleBias(nn.Module):
    def __init__(self, scale=True, scale_init=1.0, bias=True, bias_init=0.0) -> None:
        super(ScalarScaleBias, self).__init__()
        if scale:
            self.weight = Parameter(torch.Tensor(1))
        else:
            self.register_parameter("weight", None)
        if bias:
            self.bias = Parameter(torch.Tensor(1))
        else:
            self.register_parameter("bias", None)
        self.weight_init = scale_init
        self.bias_init = bias_init
        self.reset_parameters()

    # Change the default initialisation for BatchNorm
    def reset_parameters(self) -> None:
        if self.weight is not None:
            init.constant_(self.weight, self.weight_init)
        if self.bias is not None:
            init.constant_(self.bias, self.bias_init)

    def forward(self, x):
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


class ScalarBiasScale(ScalarScaleBias):
    def forward(self, x):
        if self.bias is not None:
            x = x + self.bias
        if self.weight is not None:
            x = x * self.weight
        return x
