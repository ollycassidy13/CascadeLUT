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
from functools import reduce
from os.path import realpath

import torch
import torch.nn as nn

from brevitas.core.quant import QuantType
from brevitas.core.scaling import ParameterScaling
from brevitas.nn import QuantHardTanh, QuantReLU, QuantIdentity

from pyverilator import PyVerilator

from cascadelut.quant import QuantBrevitasActivation
from cascadelut.quant import (
    LearnablePerFeatureMultiBitActivation,
    CascadeCompositeInputQuant,
    patch_cascade_layer,
)
from cascadelut.nn import (
    SparseLinearNeq,
    ScalarBiasScale,
    SupportMask,
)

def _parse_cascade_chunks(cascade_layers):
    result = []
    for cl in cascade_layers:
        if cl == 0:
            result.append([])
        elif isinstance(cl, (int, float)) and int(cl) > 0:
            result.append([int(cl) - 1])
        elif isinstance(cl, (list, tuple)):
            result.append(sorted(int(c) - 1 for c in cl))
        else:
            raise ValueError(f"Invalid cascade_layers entry: {cl!r}")
    assert result[0], "First hidden layer must consume at least one chunk (cascade_layers[0] != 0)"
    return result

class MnistCascadeModel(nn.Module):
    """
    Cascade model: each hidden layer consumes the previous layer's
    quantized output concatenated with the next chunk of raw inputs (by importance).
    The output layer consumes only the final hidden output.

    With per-layer control via `cascade_layers` (0/1 flags), a hidden layer can
    either:
    - 1: take [prev_hidden || next_chunk]
    - 0: take [prev_hidden] only (no new chunk at that layer)
    """
    def __init__(self, model_config):
        super(MnistCascadeModel, self).__init__()
        self.model_config = model_config
        self.is_cuda = model_config["cuda"]
        self.input_length = model_config["input_length"]
        self.output_length = model_config["output_length"]
        self.hidden_layers = model_config["hidden_layers"]
        self.features_per_cycle = model_config.get("features_per_cycle")
        self.importance_order = model_config.get("importance_order")
        self.dense_forward = model_config.get("dense_forward", False)
        self.full_input_mode = model_config.get("full_input_mode", False)

        assert self.features_per_cycle is not None, "features_per_cycle must be set for cascade"
        assert self.importance_order is not None, "importance_order must be set for cascade"

        initial_value = 1.33

        cl = model_config.get("cascade_layers")
        if isinstance(cl, torch.Tensor):
            cl = cl.tolist()
        self.cascade_layers = [int(v) if not isinstance(v, (list, tuple)) else list(v)
                               for v in cl]
        self.cascade_chunks = _parse_cascade_chunks(self.cascade_layers)

        all_used = sorted(set(c for layer in self.cascade_chunks for c in layer))
        num_chunks_total = (max(all_used) + 1) if all_used else 0

        chunk_features = self.input_length if self.full_input_mode \
                         else min(self.features_per_cycle, self.input_length)

        self.per_chunk_bn = nn.ModuleList()
        self.per_chunk_dropout = nn.ModuleList()
        self.per_chunk_bias = nn.ModuleList()
        self.per_chunk_quant = nn.ModuleList()
        for _ in range(num_chunks_total):
            bn_in = nn.BatchNorm1d(chunk_features)
            nn.init.constant_(bn_in.weight.data, 1)
            nn.init.constant_(bn_in.bias.data, 0)
            self.per_chunk_bn.append(bn_in)
            p = model_config.get("input_dropout")
            self.per_chunk_dropout.append(nn.Dropout(p=p) if p else nn.Identity())
            self.per_chunk_bias.append(ScalarBiasScale(scale=False, bias_init=-0.25))
            self.per_chunk_quant.append(
                LearnablePerFeatureMultiBitActivation(
                    num_features=chunk_features,
                    num_bits=model_config["input_bitwidth"],
                    initial_threshold=0.0,
                    temperature=1.0,
                    cuda=self.is_cuda,
                )
            )

        layer_list = []
        prev_out_features = 0
        imask_base = model_config.get("imask", None)
        imask_offset = 0

        for i, out_features in enumerate(self.hidden_layers):
            chunks_here = self.cascade_chunks[i]
            num_chunks_here = len(chunks_here)
            add_chunk = num_chunks_here > 0
            is_first = (i == 0)

            chunk_total_features = chunk_features * num_chunks_here
            if is_first:
                in_features = chunk_total_features          # no hidden yet
            else:
                in_features = prev_out_features + chunk_total_features

            is_support_hidden_layer = (not add_chunk) and \
                                      (model_config["support_layers"][i] == 1)
            if is_first:
                layer_fan_in_cfg = model_config["input_fanin"]
            elif add_chunk:
                layer_fan_in_cfg = model_config.get("cascade_fanin", model_config["hidden_fanin"])
            elif is_support_hidden_layer:
                layer_fan_in_cfg = model_config["support_fanin"]
            else:
                layer_fan_in_cfg = model_config["hidden_fanin"]

            bn = nn.BatchNorm1d(out_features)
            nn.init.constant_(bn.weight.data, 1)
            nn.init.constant_(bn.bias.data, 0)
            if model_config["support_layers"][i] == 1:
                output_quant = QuantBrevitasActivation(
                    QuantIdentity(bit_width=model_config["support_bitwidth"],
                                  quant_type=QuantType.INT,
                                  scaling_impl=ParameterScaling(initial_value),
                                  return_quant_tensor=False),
                    pre_transforms=[bn])
            else:
                output_quant = QuantBrevitasActivation(
                    QuantReLU(bit_width=model_config["hidden_bitwidth"],
                              quant_type=QuantType.INT,
                              scaling_impl=ParameterScaling(initial_value),
                              return_quant_tensor=False),
                    pre_transforms=[bn])

            if is_support_hidden_layer:
                imask = SupportMask(out_features, layer_fan_in_cfg)
                layer_fan_in = layer_fan_in_cfg
                layer_support = True
                layer_dense_forward = False
            else:
                use_learned = (imask_base is not None and imask_base.numel() > 0
                               and imask_offset + out_features <= imask_base.size(0)
                               and imask_base.size(1) > 0)
                if use_learned:
                    imask = imask_base[imask_offset:imask_offset + out_features,
                                       :layer_fan_in_cfg].clone().long()
                    layer_fan_in = layer_fan_in_cfg
                else:
                    layer_fan_in = layer_fan_in_cfg
                    imask = torch.zeros(out_features, layer_fan_in, dtype=torch.long)
                    for n in range(out_features):
                        for j in range(layer_fan_in):
                            imask[n, j] = (n * layer_fan_in + j) % in_features
                imask_offset += out_features
                layer_support = False
                layer_dense_forward = self.dense_forward

            if is_first and num_chunks_here == 1:
                layer_input_quant = self.per_chunk_quant[chunks_here[0]]
            elif not add_chunk:
                layer_input_quant = layer_list[-1].output_quant
            else:
                layer_input_quant = CascadeCompositeInputQuant(
                    hidden_quant=(layer_list[-1].output_quant if not is_first else None),
                    hidden_size=(prev_out_features if not is_first else 0),
                    chunk_quants=[self.per_chunk_quant[c] for c in chunks_here],
                    chunk_sizes=[chunk_features] * num_chunks_here,
                    cuda=self.is_cuda,
                )

            layer = SparseLinearNeq(
                in_features, out_features,
                input_quant=layer_input_quant,
                output_quant=output_quant,
                imask=imask,
                support=layer_support,
                dense_forward=layer_dense_forward,
                fan_in=layer_fan_in,
                width_n=model_config["width_n"],
                apply_input_quant=False,
                cuda=self.is_cuda,
                features_per_cycle=self.features_per_cycle,
                importance_order=self.importance_order,
                is_first_layer=is_first,
            )
            if isinstance(layer_input_quant, CascadeCompositeInputQuant):
                patch_cascade_layer(layer)
            layer_list.append(layer)
            prev_out_features = out_features

        bn_out = nn.BatchNorm1d(self.output_length)
        nn.init.constant_(bn_out.weight.data, 1)
        nn.init.constant_(bn_out.bias.data, 0)
        
        last_hidden_is_support = self.model_config["support_layers"][len(self.hidden_layers) - 1] == 1
        
        if last_hidden_is_support:
            output_bias_scale = ScalarBiasScale(bias_init=0)
            output_quant = QuantBrevitasActivation(
                QuantIdentity(
                    bit_width=self.model_config["output_bitwidth"],
                    quant_type=QuantType.INT,
                    scaling_impl=ParameterScaling(initial_value),
                    return_quant_tensor=False,
                ),
                pre_transforms=[bn_out],
                post_transforms=[output_bias_scale],
            )
            out_imask = SupportMask(self.output_length, self.model_config["support_fanin"])
            out_fan_in = self.model_config["support_fanin"]
            out_support = True
            out_dense_forward = False
        else:
            output_bias_scale = ScalarBiasScale(bias_init=0.33)
            output_quant = QuantBrevitasActivation(
                QuantReLU(
                    bit_width=self.model_config["output_bitwidth"],
                    quant_type=QuantType.INT,
                    scaling_impl=ParameterScaling(initial_value),
                    return_quant_tensor=False,
                ),
                pre_transforms=[bn_out],
                post_transforms=[output_bias_scale],
            )
            out_fan_in_cfg = self.model_config["hidden_fanin"]
            use_learned_out = (
                imask_base is not None
                and imask_base.numel() > 0
                and imask_offset + self.output_length <= imask_base.size(0)
                and imask_base.size(1) > 0
            )
            if use_learned_out:
                out_imask = imask_base[imask_offset:imask_offset + self.output_length, :].clone().long()
                out_fan_in = out_imask.size(1)
            else:
                out_fan_in = out_fan_in_cfg
                out_imask = torch.zeros(self.output_length, out_fan_in, dtype=torch.long)
                for neuron_idx in range(self.output_length):
                    for j in range(out_fan_in):
                        input_idx = (neuron_idx * out_fan_in + j) % prev_out_features
                        out_imask[neuron_idx, j] = input_idx
            out_support = False
            out_dense_forward = self.dense_forward
        
        out_layer = SparseLinearNeq(
            prev_out_features,
            self.output_length,
            input_quant=None,
            output_quant=output_quant,
            imask=out_imask,
            support=out_support,
            dense_forward=out_dense_forward,
            fan_in=out_fan_in,
            width_n=self.model_config["width_n"],
            apply_input_quant=False,
            cuda=self.is_cuda,
        )
        layer_list.append(out_layer)

        self.module_list = nn.ModuleList(layer_list)
        self.is_verilog_inference = False
        self.importance_order = self.importance_order
        self.is_cascade = True
        self.verilog_dir = None
        self.top_module_filename = None
        self.dut = None
        self.logfile = None
        self.verilog_latency = 0

    def verilog_inference(self, verilog_dir, top_module_filename, logfile=False, add_registers=False):
        self.verilog_dir = realpath(verilog_dir)
        self.top_module_filename = top_module_filename
        self.dut = PyVerilator.build(
            f"{self.verilog_dir}/{self.top_module_filename}",
            verilog_path=[self.verilog_dir],
            build_dir=f"{self.verilog_dir}/verilator",
        )
        self.is_verilog_inference = True
        self.logfile = logfile
        self.pipeline_depth = 1 + sum(1 for chunks in self.cascade_chunks[1:] if chunks)

    def pytorch_inference(self):
        self.is_verilog_inference = False

    def _get_chunk_bitwidth(self):
        """Return (scale, bitwidth) for a single chunk feature using layer0's input_quant."""
        return self.module_list[0].input_quant.get_scale_factor_bits(self.is_cuda)

    def _quantize_chunk_to_bin_str(self, x_sample, chunk_idx):
        """Pack one chunk into LSB-first binary string for Verilog: {feat[N-1],...,feat[0]}."""
        _, input_bitwidth = self._get_chunk_bitwidth()
        input_bitwidth = int(input_bitwidth)

        quant = self.per_chunk_quant[chunk_idx]
        raw = self._apply_chunk_transforms(x_sample, chunk_idx)

        # Restore the previous mode rather than forcing float: under lut_inference
        # these chunk quantisers are the cascade layers' input_quant, and
        # forward_incremental feeds their output straight into table_lookup,
        # which only accepts bin codes.
        was_bin = quant.is_bin_output
        quant.bin_output()
        bin_vals = quant(raw)[0]
        if not was_bin:
            quant.float_output()

        feat_bin_strs = [
            quant.get_bin_str_from_int(v, is_cuda=self.is_cuda) for v in bin_vals
        ]
        return reduce(lambda a, b: a + b, feat_bin_strs[::-1])

    def verilog_forward(self, x):
        _, input_bitwidth = self.module_list[0].input_quant.get_scale_factor_bits(self.is_cuda)
        _, output_bitwidth = self.module_list[-1].output_quant.get_scale_factor_bits(self.is_cuda)
        input_bitwidth = int(input_bitwidth)
        output_bitwidth = int(output_bitwidth)

        chunk_bits = self.features_per_cycle * input_bitwidth
        total_output_bits = self.output_length * output_bitwidth

        all_used_chunks = sorted(set(c for layer in self.cascade_chunks for c in layer))
        num_chunks = max(all_used_chunks) + 1

        output_quant = self.module_list[-1].output_quant
        self.eval()

        y = torch.zeros(x.shape[0], self.output_length)

        for sample_idx in range(x.shape[0]):
            x_i = x[sample_idx : sample_idx + 1]

            with torch.no_grad():
                y_ref = self.forward_incremental(x_i, cycle=len(self.hidden_layers) - 1)[0]

            oq_was_bin = output_quant.is_bin_output
            output_quant.bin_output()
            ys_bin = [
                output_quant.get_bin_str_from_int(v, is_cuda=self.is_cuda)
                for v in y_ref
            ]
            ysc_i = reduce(lambda a, b: a + b, ys_bin[::-1])
            if not oq_was_bin:
                output_quant.float_output()

            chunk_bin = {}
            for c in all_used_chunks:
                chunk_bin[c] = self._quantize_chunk_to_bin_str(x_i, c)

            self.dut.io.rst = 1
            self.dut.io.clk = 1
            self.dut.io.clk = 0
            self.dut.io.rst = 0

            # Each chunk c is clocked in on cycle c; unused slots (gaps) are driven 0 and treated as no-ops.
            for c in range(num_chunks):
                bin_str = chunk_bin.get(c, "0" * chunk_bits)
                self.dut["input_chunk"] = int(bin_str, 2)
                self.dut.io.clk = 1
                self.dut.io.clk = 0

            res = self.dut["output_data"]
            result = f"{res:0{total_output_bits}b}"
            expected = f"{int(ysc_i, 2):0{total_output_bits}b}"

            if result != expected:
                print(f"[MISMATCH] sample={sample_idx}")
                print(f"  expected : {expected}")
                print(f"  got      : {result}")
                for c in all_used_chunks:
                    print(f"  chunk[{c}] : {chunk_bin[c]}")
            assert result == expected, (
                f"Verilog/PyTorch mismatch at sample {sample_idx}: "
                f"expected {expected}, got {result}"
            )

            res_split = [
                result[j : j + output_bitwidth]
                for j in range(0, total_output_bits, output_bitwidth)
            ][::-1]
            y[sample_idx, :] = torch.tensor(
                [int(s, 2) for s in res_split], dtype=torch.float32
            )

            if self.logfile is not None:
                all_chunks_str = "".join(
                    chunk_bin.get(c, "0" * chunk_bits) for c in range(num_chunks)
                )
                with open(self.logfile, "a") as f:
                    f.write(f"{all_chunks_str}{int(ysc_i, 2):0{total_output_bits}b}\n")

        return y

    def _get_chunk(self, x, chunk_idx: int):
        if self.full_input_mode:
            return x
        B = self.features_per_cycle
        start = chunk_idx * B
        end = min((chunk_idx + 1) * B, self.input_length)
        idx = self.importance_order[start:end]
        if isinstance(idx, torch.Tensor):
            idx = idx.long()
        return x[:, idx]

    def _apply_chunk_transforms(self, x, chunk_idx: int):
        """Apply dropout → BN → bias for a given chunk index, return raw float chunk."""
        chunk = self._get_chunk(x, chunk_idx)
        if self.is_cuda:
            chunk = chunk.cuda()
        chunk = self.per_chunk_dropout[chunk_idx](chunk)
        chunk = self.per_chunk_bn[chunk_idx](chunk)
        chunk = self.per_chunk_bias[chunk_idx](chunk)
        return chunk

    def forward_incremental(self, x, cycle: int = 0):
        prev = None
        for i in range(min(cycle + 1, len(self.hidden_layers))):
            chunks_here = self.cascade_chunks[i]

            if not chunks_here:
                layer_input = prev
            else:
                chunk_parts = []
                for c in chunks_here:
                    raw = self._apply_chunk_transforms(x, c)
                    chunk_q = self.per_chunk_quant[c](raw)
                    chunk_parts.append(chunk_q)

                if prev is None:
                    layer_input = torch.cat(chunk_parts, dim=1) if len(chunk_parts) > 1 \
                                  else chunk_parts[0]
                else:
                    layer_input = torch.cat([prev] + chunk_parts, dim=1)

            prev = self.module_list[i](layer_input)

        return self.module_list[-1](prev)

    def forward(self, x):
        if self.is_verilog_inference:
            return self.verilog_forward(x)
        return self.forward_incremental(x, cycle=len(self.hidden_layers) - 1)

class MnistCascadeLutModel(MnistCascadeModel):
    pass


class MnistCascadeVerilogModel(MnistCascadeModel):
    pass