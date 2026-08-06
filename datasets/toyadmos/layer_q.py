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
from torch.utils.data import DataLoader, TensorDataset
from pyverilator import PyVerilator

from dataset import load_dataset


def compute_layer_q_truth_tables(
    model: nn.Module,
    feature_min_arr: np.ndarray = None,
    feature_max_arr: np.ndarray = None,
    output_bits: Optional[int] = None,
) -> List[List[Tuple[int, List[int]]]]:
    model.eval()

    all_used = sorted(set(c for layer in model.cascade_chunks for c in layer))
    if not all_used:
        return []
    num_chunks = max(all_used) + 1
    fpc = model.features_per_cycle
    imp = model.importance_order
    if isinstance(imp, torch.Tensor):
        imp = imp.cpu().numpy()

    if output_bits is None:
        q0 = model.per_chunk_quant[0]
        output_bits = int(getattr(q0, "num_bits", 1))

    if feature_min_arr is None:
        feature_min_arr = np.zeros(model.input_length)
    if feature_max_arr is None:
        feature_max_arr = np.ones(model.input_length)

    truth_tables: List[List[Tuple[int, List[int]]]] = []

    u8 = torch.arange(256, dtype=torch.float32)

    with torch.no_grad():
        for c in range(num_chunks):
            bn_mod = model.per_chunk_bn[c].eval()
            bias_mod = model.per_chunk_bias[c]
            quant_mod = model.per_chunk_quant[c].eval()

            start = c * fpc
            end = min((c + 1) * fpc, model.input_length)
            csz = end - start
            global_idxs = [int(imp[start + j]) for j in range(csz)]

            x_batch = torch.zeros(256, csz)
            for j in range(csz):
                g_j = global_idxs[j]
                fmin_j = float(feature_min_arr[g_j])
                fmax_j = float(feature_max_arr[g_j])
                fr_j = max(fmax_j - fmin_j, 1e-9)
                x_batch[:, j] = u8 / 255.0 * fr_j + fmin_j

            if model.is_cuda:
                x_batch = x_batch.cuda()

            x_bn = bn_mod(x_batch)
            x_b = bias_mod(x_bn)

            was_bin = quant_mod.is_bin_output
            quant_mod.bin_output()
            q_out = quant_mod(x_b)
            q_int = q_out.long().cpu()
            if not was_bin:
                quant_mod.float_output()

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
        self.truth_tables = truth_tables
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
            csz = len(chunk_luts)
            start = c * self.features_per_cycle
            feat_idxs = self.imp[start:start + csz].astype(int)
            chunk_raw = x_np[:, feat_idxs]
            chunk_q = np.stack(
                [chunk_luts[j][chunk_raw[:, j]] for j in range(csz)],
                axis=1,
            )
            chunks_out.append(chunk_q)

        result = np.concatenate(chunks_out, axis=1)
        return torch.from_numpy(result.astype(np.float32))


def _neuron_verilog(
    module_name: str,
    truth_vec: List[int],
    output_bits: int = 1,
) -> str:
    if output_bits == 1:
        entries = "".join(
            f"\t\t\t8'h{v:02x}: M1r = 1'b{truth_vec[v]};\n"
            for v in range(256)
        )
        default_val = "1'b0"
    else:
        entries = "".join(
            f"\t\t\t8'h{v:02x}: M1r = {output_bits}'d{truth_vec[v]};\n"
            for v in range(256)
        )
        default_val = f"{output_bits}'d0"

    return (
        f"module {module_name} "
        f"( input [7:0] M0, output [{output_bits-1}:0] M1 );\n\n"
        f"\t(*rom_style = \"distributed\" *) reg [{output_bits-1}:0] M1r;\n"
        f"\tassign M1 = M1r;\n"
        f"\talways @ (M0) begin\n"
        f"\t\tcase (M0)\n"
        f"{entries}"
        f"\t\tdefault: M1r = {default_val};\n"
        f"\t\tendcase\n"
        f"\tend\n"
        f"endmodule\n"
    )


def generate_layer_q_verilog(
    truth_tables: List[List[Tuple[int, List[int]]]],
    output_dir: str,
    output_bits: int = 1,
) -> List[Tuple[int, int, int]]:
    os.makedirs(output_dir, exist_ok=True)
    dims: List[Tuple[int, int, int]] = []

    for c, chunk_truth in enumerate(truth_tables):
        csz = len(chunk_truth)
        total_in_bits = csz * 8
        total_out_bits = csz * output_bits
        prefix = f"layer_q_{c}"

        body = (
            f"module {prefix} "
            f"(input [{total_in_bits-1}:0] M0, "
            f"output [{total_out_bits-1}:0] M1);\n\n"
        )

        for j, (_gidx, truth_vec) in enumerate(chunk_truth):
            nname = f"{prefix}_N{j}"
            with open(f"{output_dir}/{nname}.v", "w") as fh:
                fh.write(_neuron_verilog(nname, truth_vec, output_bits=output_bits))
            in_msb = j * 8 + 7
            in_lsb = j * 8
            out_msb = j * output_bits + output_bits - 1
            out_lsb = j * output_bits
            body += (
                f"wire [7:0] {nname}_w = M0[{in_msb}:{in_lsb}];\n"
                f"{nname} {nname}_inst (.M0({nname}_w), .M1(M1[{out_msb}:{out_lsb}]));\n\n"
            )

        body += "endmodule\n"
        with open(f"{output_dir}/{prefix}.v", "w") as fh:
            fh.write(body)

        dims.append((total_in_bits, total_out_bits, output_bits))
        print(
            f"  [layer_q] Generated {prefix}.v  "
            f"({csz} LUTs, 8b/{output_bits}b, "
            f"{total_in_bits}b in → {total_out_bits}b out)"
        )

    return dims


def build_uint8_test_loader(
    dset_path: str,
    batch_size: int,
    feature_min_arr: np.ndarray,
    feature_max_arr: np.ndarray,
    split: str = "test",
):
    X_train, Y_train, X_test, Y_test = load_dataset(dset_path)

    X, Y = (X_train, Y_train) if split == "train" else (X_test, Y_test)
    frange_arr = np.maximum(feature_max_arr - feature_min_arr, 1e-9)
    X_uint8 = torch.from_numpy(
        (((X - feature_min_arr) / frange_arr) * 255.0)
        .clip(0, 255)
        .astype(np.uint8)
    ).float()
    Y_tensor = torch.from_numpy(Y.astype(np.int64)).long()

    return DataLoader(TensorDataset(X_uint8, Y_tensor), batch_size=batch_size, shuffle=False)


def _float_to_uint8(
    x: torch.Tensor,
    feature_min_arr: np.ndarray,
    feature_max_arr: np.ndarray,
) -> torch.Tensor:
    fmin = torch.from_numpy(feature_min_arr).float()
    fmax = torch.from_numpy(feature_max_arr).float()
    frange = torch.clamp(fmax - fmin, min=1e-9)
    return ((x - fmin) / frange * 255.0).clamp(0.0, 255.0).round().long()


def _build_raw_words(
    x_uint8_sample: torch.Tensor,
    chunk_idx: int,
    features_per_cycle: int,
    importance_order: torch.Tensor,
    hw_bandwidth: int,
) -> list:
    fpc = features_per_cycle
    start = chunk_idx * fpc
    end = min((chunk_idx + 1) * fpc, len(importance_order))
    csz = end - start

    imp = importance_order.cpu().numpy().astype(int)
    x_np = x_uint8_sample.cpu().numpy()
    packed = 0
    for j in range(csz):
        packed |= (int(x_np[imp[start + j]]) & 0xFF) << (j * 8)

    total_bits = csz * 8
    load_cpc = math.ceil(total_bits / hw_bandwidth)
    words = []
    for k in range(load_cpc):
        lo = k * hw_bandwidth
        hi = min(lo + hw_bandwidth, total_bits)
        mask = (1 << (hi - lo)) - 1
        words.append((packed >> lo) & mask)
    return words


def verilog_forward_layer_q_v2(self, x: torch.Tensor) -> torch.Tensor:
    _, out_bw = self.module_list[-1].output_quant.get_scale_factor_bits(self.is_cuda)
    out_bw = int(out_bw)
    total_out = self.output_length * out_bw
    out_quant = self.module_list[-1].output_quant

    hw_bw = self.hw_bandwidth
    fpc = self.features_per_cycle

    self.eval()
    y = torch.zeros(x.shape[0], self.output_length)
    x_uint8 = _float_to_uint8(x.cpu(), self.feature_min_arr, self.feature_max_arr)

    lq_sim = getattr(self, 'layer_q_simulator', None)

    mismatches = 0

    for sample_idx in range(x.shape[0]):
        x_i = x[sample_idx : sample_idx + 1]
        x_u8 = x_uint8[sample_idx]

        with torch.no_grad():
            if lq_sim is not None:
                q_out = lq_sim(x_u8.unsqueeze(0))
                if self.is_cuda:
                    q_out = q_out.cuda()
                prev = None
                for i, hidden_layer in enumerate(self.module_list[:-1]):
                    chunks_here = self.cascade_chunks[i]
                    if not chunks_here:
                        layer_input = prev
                    else:
                        parts = [q_out[:, c * fpc : (c + 1) * fpc] for c in chunks_here]
                        if prev is None:
                            layer_input = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
                        else:
                            layer_input = torch.cat([prev] + parts, dim=1)
                    prev = hidden_layer(layer_input)
                y_ref = self.module_list[-1](prev)[0]
            else:
                y_ref = self.forward_incremental(x_i, cycle=len(self.hidden_layers) - 1)[0]

        out_quant.bin_output()
        expected = "".join(
            out_quant.get_bin_str_from_int(v, is_cuda=self.is_cuda)
            for v in reversed(y_ref)
        )
        out_quant.float_output()

        self.dut.io.rst = 1
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

        res = self.dut["output_data"]
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
    feature_min_arr: np.ndarray = None,
    feature_max_arr: np.ndarray = None,
    layer_q_truth_tables=None,
    logfile=None,
):
    self.verilog_dir = os.path.realpath(verilog_dir)
    self.top_module_filename = top_module_filename
    self.dut = PyVerilator.build(
        f"{self.verilog_dir}/{self.top_module_filename}",
        verilog_path=[self.verilog_dir],
        build_dir=f"{self.verilog_dir}/verilator",
    )
    self.is_verilog_inference = True
    self.logfile = logfile
    self.hw_bandwidth = hw_bandwidth
    self.num_chunks = num_chunks
    self.feature_min_arr = feature_min_arr if feature_min_arr is not None else np.zeros(1)
    self.feature_max_arr = feature_max_arr if feature_max_arr is not None else np.ones(1)
    self.layer_q_simulator = (
        LayerQLutSimulator(layer_q_truth_tables, self.features_per_cycle, self.importance_order)
        if layer_q_truth_tables is not None else None
    )

    fpc = self.features_per_cycle
    load_cpc = math.ceil(fpc * 8 / hw_bandwidth)
    total_cyc = num_chunks * load_cpc

    self.verilog_forward = types.MethodType(verilog_forward_layer_q_v2, self)

    print(
        f"[Layer-Q sim] DUT: {top_module_filename}\n"
        f"  hw_bandwidth={hw_bandwidth}  features_per_cycle={fpc}\n"
        f"  load_cpc={load_cpc}  num_chunks={num_chunks}\n"
        f"  raw_buf_bits/chunk={(load_cpc-1)*hw_bandwidth}  "
        f"  Cycles/sample = {num_chunks}x{load_cpc} = {total_cyc}  "
    )
