#  This file is part of CascadeLUT.
#
#  CascadeLUT is a derivative work based on LogicNets,
#  which is licensed under the Apache License 2.0.

from __future__ import annotations

import math
import os
import types
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from pyverilator import PyVerilator


def compute_layer_q_truth_tables(
    model: nn.Module,
    dataset_mean: float = 0.1307,
    dataset_std: float = 0.3081,
) -> List[List[Tuple[int, List[int]]]]:
    model.eval()

    all_used = sorted(set(c for layer in model.cascade_chunks for c in layer))
    if not all_used:
        return []
    num_chunks = max(all_used) + 1
    fpc        = model.features_per_cycle
    imp        = model.importance_order
    if isinstance(imp, torch.Tensor):
        imp = imp.cpu().numpy()

    truth_tables: List[List[Tuple[int, List[int]]]] = []

    u8     = torch.arange(256, dtype=torch.float32)
    x_norm = (u8 / 255.0 - dataset_mean) / dataset_std

    with torch.no_grad():
        for c in range(num_chunks):
            bn_mod    = model.per_chunk_bn[c].eval()
            bias_mod  = model.per_chunk_bias[c]
            quant_mod = model.per_chunk_quant[c].eval()

            start       = c * fpc
            end         = min((c + 1) * fpc, model.input_length)
            csz         = end - start
            global_idxs = [int(imp[start + j]) for j in range(csz)]

            x_batch = x_norm.unsqueeze(1).expand(256, csz).clone()
            if model.is_cuda:
                x_batch = x_batch.cuda()

            x_bn  = bn_mod(x_batch)
            x_b   = bias_mod(x_bn)

            quant_mod.float_output()
            q_out = quant_mod(x_b)
            q_int = q_out.long().cpu()

            chunk_truth = [
                (global_idxs[j], q_int[:, j].tolist())
                for j in range(csz)
            ]
            truth_tables.append(chunk_truth)

    return truth_tables


class LayerQLutSimulator:
    def __init__(
        self,
        truth_tables: List[List[Tuple[int, List[int]]]],
        features_per_cycle: int,
        importance_order,
    ):
        self.truth_tables       = truth_tables
        self.features_per_cycle = features_per_cycle
        self.imp = (
            importance_order.cpu().numpy()
            if isinstance(importance_order, torch.Tensor)
            else np.asarray(importance_order)
        )
        self._luts = [
            [np.array(truth_vec, dtype=np.float32) for _, truth_vec in chunk]
            for chunk in truth_tables
        ]

    def __call__(self, x_uint8: torch.Tensor) -> torch.Tensor:
        x_np = x_uint8.cpu().numpy().astype(np.uint8)
        chunks_out = []
        for c, chunk_luts in enumerate(self._luts):
            csz       = len(chunk_luts)
            start     = c * self.features_per_cycle
            feat_idxs = self.imp[start:start + csz].astype(int)
            chunk_raw = x_np[:, feat_idxs]
            chunk_q   = np.stack(
                [chunk_luts[j][chunk_raw[:, j]] for j in range(csz)],
                axis=1,
            )
            chunks_out.append(chunk_q)

        result = np.concatenate(chunks_out, axis=1)
        return torch.from_numpy(result.astype(np.float32))


def _neuron_verilog(module_name: str, truth_vec: List[int]) -> str:
    entries = "".join(
        f"\t\t\t8'h{v:02x}: M1r = 1'b{truth_vec[v]};\n"
        for v in range(256)
    )
    return (
        f"module {module_name} ( input [7:0] M0, output [0:0] M1 );\n\n"
        f"\t(*rom_style = \"distributed\" *) reg [0:0] M1r;\n"
        f"\tassign M1 = M1r;\n"
        f"\talways @ (M0) begin\n"
        f"\t\tcase (M0)\n"
        f"{entries}"
        f"\t\tdefault: M1r = 1'b0;\n"
        f"\t\tendcase\n"
        f"\tend\n"
        f"endmodule\n"
    )


def generate_layer_q_verilog(
    truth_tables: List[List[Tuple[int, List[int]]]],
    output_dir: str,
) -> List[Tuple[int, int]]:
    os.makedirs(output_dir, exist_ok=True)
    dims: List[Tuple[int, int]] = []

    for c, chunk_truth in enumerate(truth_tables):
        csz            = len(chunk_truth)
        total_in_bits  = csz * 8
        total_out_bits = csz
        prefix         = f"layer_q_{c}"

        body = (
            f"module {prefix} "
            f"(input [{total_in_bits-1}:0] M0, "
            f"output [{total_out_bits-1}:0] M1);\n\n"
        )

        for j, (_gidx, truth_vec) in enumerate(chunk_truth):
            nname = f"{prefix}_N{j}"
            with open(f"{output_dir}/{nname}.v", "w") as fh:
                fh.write(_neuron_verilog(nname, truth_vec))
            msb = j * 8 + 7
            lsb = j * 8
            body += f"wire [7:0] {nname}_w = M0[{msb}:{lsb}];\n"
            body += f"{nname} {nname}_inst (.M0({nname}_w), .M1(M1[{j}]));\n\n"

        body += "endmodule\n"
        with open(f"{output_dir}/{prefix}.v", "w") as fh:
            fh.write(body)

        dims.append((total_in_bits, total_out_bits))
        print(f"  [layer_q] Generated {prefix}.v  "
              f"({csz} LUTs, {total_in_bits}b in → {total_out_bits}b out)")

    return dims


class Uint8MnistTransform:
    def __call__(self, img):
        arr = np.array(img, dtype=np.uint8).flatten()
        return torch.from_numpy(arr).float()

    @staticmethod
    def to_uint8(x_float: torch.Tensor) -> torch.Tensor:
        pixel_float = x_float * 0.3081 + 0.1307
        return (pixel_float * 255.0).clamp(0, 255).round().to(torch.uint8)


def build_uint8_test_loader(data_root: str, batch_size: int, split: str = "test"):
    dataset = datasets.MNIST(
        data_root,
        download=False,
        train=(split == "train"),
        transform=Uint8MnistTransform(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _float_to_uint8(
    x_norm: torch.Tensor,
    mean: float = 0.1307,
    std:  float = 0.3081,
) -> torch.Tensor:
    pixel = (x_norm * std + mean) * 255.0
    return pixel.clamp(0.0, 255.0).round().long()


def _build_raw_words(
    x_uint8_sample: torch.Tensor,
    chunk_idx: int,
    features_per_cycle: int,
    importance_order: torch.Tensor,
    hw_bandwidth: int,
) -> list:
    fpc   = features_per_cycle
    start = chunk_idx * fpc
    end   = min((chunk_idx + 1) * fpc, len(importance_order))
    csz   = end - start

    imp    = importance_order.cpu().numpy().astype(int)
    x_np   = x_uint8_sample.cpu().numpy()
    packed = 0
    for j in range(csz):
        packed |= (int(x_np[imp[start + j]]) & 0xFF) << (j * 8)

    total_bits = csz * 8
    load_cpc   = math.ceil(total_bits / hw_bandwidth)
    words = []
    for k in range(load_cpc):
        lo   = k * hw_bandwidth
        hi   = min(lo + hw_bandwidth, total_bits)
        mask = (1 << (hi - lo)) - 1
        words.append((packed >> lo) & mask)
    return words


def verilog_forward_layer_q_v2(self, x: torch.Tensor) -> torch.Tensor:
    _, out_bw = self.module_list[-1].output_quant.get_scale_factor_bits(self.is_cuda)
    out_bw    = int(out_bw)
    total_out = self.output_length * out_bw
    out_quant = self.module_list[-1].output_quant

    hw_bw = self.hw_bandwidth
    fpc   = self.features_per_cycle

    self.eval()
    y       = torch.zeros(x.shape[0], self.output_length)
    x_uint8 = _float_to_uint8(x.cpu(), self.dataset_mean, self.dataset_std)

    mismatches = 0

    for sample_idx in range(x.shape[0]):
        x_i  = x[sample_idx : sample_idx + 1]
        x_u8 = x_uint8[sample_idx]

        with torch.no_grad():
            y_ref = self.forward_incremental(x_i, cycle=len(self.hidden_layers) - 1)[0]
        out_quant.bin_output()
        expected = "".join(
            out_quant.get_bin_str_from_int(v, is_cuda=self.is_cuda)
            for v in reversed(y_ref)
        )
        out_quant.float_output()

        self.dut.io.rst       = 1
        self.dut["raw_input"] = 0
        self.dut.io.clk = 1
        self.dut.io.clk = 0
        self.dut.io.rst = 0

        for c in range(self.num_chunks):
            words = _build_raw_words(x_u8, c, fpc, self.importance_order, hw_bw)
            for word in words:
                self.dut["raw_input"] = word
                self.dut.io.clk = 1
                self.dut.io.clk = 0

        res    = self.dut["output_data"]
        result = f"{res:0{total_out}b}"

        if result != expected:
            mismatches += 1
            if mismatches <= 5:
                print(f"[MISMATCH sample={sample_idx}]")
                print(f"  expected: {expected}")
                print(f"  got     : {result}")

        parts = [result[j:j+out_bw] for j in range(0, total_out, out_bw)][::-1]
        y[sample_idx, :] = torch.tensor(
            [int(s, 2) for s in parts], dtype=torch.float32
        )

        if self.logfile is not None:
            with open(self.logfile, "a") as f:
                f.write(f"{expected}\n")

    if mismatches > 0:
        print(
            f"[Layer-Q sim] {mismatches}/{x.shape[0]} mismatches "
            f"({100*mismatches/x.shape[0]:.2f}%)"
        )
    return y


def verilog_inference_layer_q(
    self,
    verilog_dir: str,
    top_module_filename: str,
    hw_bandwidth: int,
    num_chunks: int,
    dataset_mean: float = 0.1307,
    dataset_std:  float = 0.3081,
    logfile=None,
):
    self.verilog_dir         = os.path.realpath(verilog_dir)
    self.top_module_filename = top_module_filename
    self.dut = PyVerilator.build(
        f"{self.verilog_dir}/{self.top_module_filename}",
        verilog_path=[self.verilog_dir],
        build_dir=f"{self.verilog_dir}/verilator",
    )
    self.is_verilog_inference = True
    self.logfile              = logfile
    self.hw_bandwidth         = hw_bandwidth
    self.num_chunks           = num_chunks
    self.dataset_mean         = dataset_mean
    self.dataset_std          = dataset_std

    fpc       = self.features_per_cycle
    load_cpc  = math.ceil(fpc * 8 / hw_bandwidth)
    total_cyc = num_chunks * load_cpc

    self.verilog_forward = types.MethodType(verilog_forward_layer_q_v2, self)

    print(
        f"[Layer-Q sim] DUT: {top_module_filename}\n"
        f"  hw_bandwidth={hw_bandwidth}  features_per_cycle={fpc}\n"
        f"  load_cpc={load_cpc}  num_chunks={num_chunks}\n"
        f"  raw_buf_bits/chunk={(load_cpc-1)*hw_bandwidth}  "
        f"  Cycles/sample = {num_chunks}x{load_cpc} = {total_cyc}  "
    )
