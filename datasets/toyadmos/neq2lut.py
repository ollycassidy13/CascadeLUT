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

import os
import math
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from cascadelut.nn import (
    module_list_to_verilog_module,
    lut_inference,
    generate_truth_tables,
    SparseLinearNeq,
)
from train import configs, model_config, test
from models import ToyCarCascadeModel, ToyCarCascadeLutModel
from dataset import load_dataset
from cascadelut.synthesis import (
    synthesize_and_get_resource_counts,
    get_power_analysis_fast,
)

from layer_q import (
    compute_layer_q_truth_tables,
    generate_layer_q_verilog,
    LayerQLutSimulator,
    verilog_inference_layer_q,
)
from cascadelut.verilog_layer_q import generate_cascade_top_v2

other_options = {
    "cuda": None,
    "device": 1,
    "log_dir": None,
    "checkpoint": None,
    "imask": None,
    "add_registers": False,
    "importance_order": None,
}


def test_layer_q_lut_path(
    model,
    truth_tables,
    test_loader_uint8,
    cuda: bool,
) -> float:
    print("\n[Layer-Q] Validating uint8 → LUT simulation path...")
    model.eval()

    lut_sim = LayerQLutSimulator(
        truth_tables,
        features_per_cycle=model.features_per_cycle,
        importance_order=model.importance_order,
    )

    chunk_offsets = {}
    offset = 0
    for c_idx, chunk_truth in enumerate(truth_tables):
        chunk_offsets[c_idx] = (offset, offset + len(chunk_truth))
        offset += len(chunk_truth)

    correct = 0
    total = 0

    with torch.no_grad():
        for x_uint8, y in test_loader_uint8:
            x_q = lut_sim(x_uint8)
            x_in = x_q.cuda() if cuda else x_q
            prev = None
            for i, layer in enumerate(model.module_list[:-1]):
                chunks_here = model.cascade_chunks[i]
                if chunks_here:
                    chunk_parts = [
                        x_in[:, chunk_offsets[c][0]:chunk_offsets[c][1]]
                        for c in chunks_here
                    ]
                    if prev is None:
                        layer_in = (
                            torch.cat(chunk_parts, dim=1)
                            if len(chunk_parts) > 1 else chunk_parts[0]
                        )
                    else:
                        layer_in = torch.cat([prev] + chunk_parts, dim=1)
                else:
                    layer_in = prev
                prev = layer.lut_forward(layer_in)

            out = model.module_list[-1].lut_forward(prev)
            preds = out.argmax(dim=1).cpu()
            correct += (preds == y).sum().item()
            total   += x_uint8.size(0)

    accuracy = correct / total
    print(f"[Layer-Q] uint8 → LUT accuracy: {accuracy:.6f}  ({correct}/{total})")
    return accuracy


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Convert a trained CascadeLUT model to synthesisable Verilog"
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=configs.keys(),
        default="hdr-5l-0.1",
        help="Specific the neural network model to use (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Batch size for evaluation (default: %(default)s)",
    )
    parser.add_argument(
        "--input-bitwidth",
        type=int,
        default=None,
        help="Bitwidth to use at the input (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden-bitwidth",
        type=int,
        default=None,
        help="Bitwidth to use for activations in hidden layers (default: %(default)s)",
    )
    parser.add_argument(
        "--output-bitwidth",
        type=int,
        default=None,
        help="Bitwidth to use at the output (default: %(default)s)",
    )
    parser.add_argument(
        "--support_bitwidth",
        type=int,
        default=None,
        metavar="",
        help="Bitwidth to use at the support layer (default: %(default)s)",
    )
    parser.add_argument(
        "--input-fanin",
        type=int,
        default=None,
        help="Fanin to use at the input (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden-fanin",
        type=int,
        default=None,
        help="Fanin to use for the hidden layers (default: %(default)s)",
    )
    parser.add_argument(
        "--support_layers",
        nargs="+",
        type=int,
        default=None,
        metavar="",
        help="A boolean list of support layers (default: %(default)s)",
    )
    parser.add_argument(
        "--output-fanin",
        type=int,
        default=None,
        help="Fanin to use at the output (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden-layers",
        nargs="+",
        type=int,
        default=None,
        help="A list of hidden layer neuron sizes (default: %(default)s)",
    )
    parser.add_argument(
        "--width_n",
        type=int,
        default=None,
        metavar="",
        help="Width of sub-network(N) (default: %(default)s)",
    )
    parser.add_argument(
        "--clock-period",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset to use for evaluation (default: %(default)s)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="0",
        help="A location to store the log output of the training run and the output model (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=int,
        required=True,
        help="Device_id for GPU (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="The checkpoint file which contains the model weights",
    )
    parser.add_argument(
        "--imask",
        type=str,
        required=True,
        help="The checkpoint file which contains the model weights",
    )
    parser.add_argument(
        "--add-registers",
        action="store_true",
        default=False,
        help="Add registers between each layer in generated verilog (default: %(default)s)",
    )
    parser.add_argument(
        "--support_fanin",
        type=int,
        default=None,
        metavar="",
    )
    parser.add_argument(
        "--importance-order",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--cascade-layers",
        nargs="+",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--cascade-fanin",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--features-per-cycle",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use-layer-q",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--hw-bandwidth",
        type=int,
        default=112,
        metavar="N",
    )
    parser.add_argument(
        "--skip-layer-q-sim",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    defaults = configs[args.arch]
    options = vars(args)
    del options["arch"]

    config = {}
    for k in options:
        config[k] = (
            options[k]
            if options[k] is not None or k == "importance_order"
            else defaults.get(k)
        )

    os.makedirs(config["log_dir"], exist_ok=True)

    use_layer_q = config["use_layer_q"]
    hw_bandwidth = config["hw_bandwidth"]
    input_bitwidth = config.get("input_bitwidth", 1)
    
    if use_layer_q:
        fpc_configured = (
            config.get("features_per_cycle") or defaults.get("features_per_cycle", 112)
        )
        fpc_bits = fpc_configured * 8
        if hw_bandwidth > fpc_bits:
            raise ValueError(
                f"--hw-bandwidth ({hw_bandwidth}) must be ≤ "
                f"features_per_cyclex8 = {fpc_configured}x8 = {fpc_bits}"
            )
        load_cpc = math.ceil(fpc_bits / hw_bandwidth)
        print(f"\n[Layer-Q] hw_bandwidth={hw_bandwidth}  "
              f"features_per_cycle={fpc_configured} features  "
              f"fpc_bits={fpc_bits}  LOAD_CPC={load_cpc}")
    else:
        fpc_configured = (
            config.get("features_per_cycle") or defaults.get("features_per_cycle", 112)
        )
        chunk_bits = fpc_configured * input_bitwidth
        if chunk_bits > hw_bandwidth:
            stall_ratio = (chunk_bits + hw_bandwidth - 1) // hw_bandwidth
            print(f"\nhw_bandwidth={hw_bandwidth}  "
                  f"features_per_cycle={fpc_configured} features x {input_bitwidth}b = {chunk_bits}b  "
                  f"→ {stall_ratio} stalls needed per cycle")
        else:
            print(f"\nhw_bandwidth={hw_bandwidth}  "
                  f"features_per_cycle={fpc_configured} features x {input_bitwidth}b = {chunk_bits}b  "
                  f"(no stalls needed)")

    # Split up configuration options to be more understandable
    model_cfg = {k: config[k] for k in model_config if k in config}
    options_cfg = {k: config[k] for k in other_options if k in config}
    model_cfg["cuda"] = options_cfg["cuda"]

    if options_cfg["cuda"]:
        torch.cuda.set_device(options_cfg["device"])

    X_train, y_train, X_test, y_test = load_dataset("./data/")

    feature_min_arr = X_train.min(axis=0)
    feature_max_arr = X_train.max(axis=0)
    print(f"\nPer-feature range (from training set):")
    print(f"  Global min of per-feature mins: {feature_min_arr.min():.6f}")
    print(f"  Global max of per-feature maxs: {feature_max_arr.max():.6f}")

    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test.astype(np.int64)).long()

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    if use_layer_q:
        frange_arr = np.maximum(feature_max_arr - feature_min_arr, 1e-9)
        X_test_uint8 = torch.from_numpy(
            (((X_test - feature_min_arr) / frange_arr) * 255.0)
            .clip(0, 255)
            .astype(np.uint8)
        ).float()
        test_dataset_uint8 = TensorDataset(X_test_uint8, y_test_tensor)
        test_loader_uint8 = DataLoader(
            test_dataset_uint8, batch_size=config["batch_size"], shuffle=False
        )

    input_length = len(X_train[0])
    output_length = len(set(y_train))
    model_cfg["input_length"] = input_length
    model_cfg["output_length"] = output_length
    model_cfg["imask"] = torch.load(
        options_cfg["imask"],
        map_location=(
            f"cuda:{options_cfg['device']}" if options_cfg["cuda"] else "cpu"
        ),
    )
    model_cfg["dense_forward"] = False

    print("\nLoading CASCADE model...")
    importance_path = options_cfg["imask"].replace(
        "imask_cascade.pth", "importance_order.pth"
    )
    if os.path.exists(importance_path):
        model_cfg["importance_order"] = torch.load(
            importance_path, map_location="cpu"
        ).long()
        print(f"  Loaded importance order from {importance_path}")
    else:
        model_cfg["importance_order"] = torch.arange(input_length, dtype=torch.long)
    model_cfg["features_per_cycle"] = config.get("features_per_cycle", 112)
    model = ToyCarCascadeModel(model_cfg)

    if options_cfg["cuda"]:
        model.cuda()

    checkpoint = torch.load(
        options_cfg["checkpoint"],
        map_location=(
            f"cuda:{options_cfg['device']}" if options_cfg["cuda"] else "cpu"
        ),
    )
    model.load_state_dict(checkpoint["model_dict"])

    print("\n" + "=" * 70)
    print("1. PyTorch float inference")
    print("=" * 70)
    baseline_accuracy = test(model, test_loader, cuda=options_cfg["cuda"])
    print(f"   Baseline accuracy: {baseline_accuracy:.6f}")

    lut_model = ToyCarCascadeLutModel(model_cfg)
    if options_cfg["cuda"]:
        lut_model.cuda()
    lut_model.load_state_dict(checkpoint["model_dict"])

    print("\n" + "=" * 70)
    print("2. NEQ → LUT truth table generation")
    print("=" * 70)
    generate_truth_tables(lut_model, verbose=True)
    lut_inference(lut_model)
    lut_accuracy = test(lut_model, test_loader, cuda=options_cfg["cuda"])
    print(f"   LUT-based accuracy: {lut_accuracy:.6f}")

    layer_q_truth_tables = None
    if use_layer_q:
        print("\n" + "=" * 70)
        print("3. Layer-Q truth table computation")
        print("=" * 70)
        layer_q_truth_tables = compute_layer_q_truth_tables(
            lut_model,
            feature_min_arr=feature_min_arr,
            feature_max_arr=feature_max_arr,
        )
        print(f"   Computed truth tables for {len(layer_q_truth_tables)} chunks")
        for c, chunk in enumerate(layer_q_truth_tables):
            ones = sum(b for _, tv in chunk for b in tv)
            print(f"     chunk {c}: {len(chunk)} LUTs, "
                  f"avg activation = {ones/(len(chunk)*256):.3f}")

        if not config.get("skip_layer_q_sim", False):
            print("\n" + "=" * 70)
            print("4. Layer-Q software LUT simulation")
            print("=" * 70)
            test_layer_q_lut_path(
                lut_model, layer_q_truth_tables, test_loader_uint8,
                cuda=options_cfg["cuda"],
            )
    log_dir = options_cfg["log_dir"]
    print("\n" + "=" * 70)
    print(f"5. Verilog generation → {log_dir}/")
    print("=" * 70)

    cascade_config = {
        "enabled": True,
        "features_per_cycle": config.get("features_per_cycle", 112),
        "cascade_chunks": lut_model.cascade_chunks,
        "cascade_layers": config.get("cascade_layers"),
    }

    if use_layer_q and layer_q_truth_tables is not None:
        print("\n[Layer-Q] Step A: generating feature L-LUT Verilog...")
        layer_q_dims = generate_layer_q_verilog(
            layer_q_truth_tables, log_dir,
            output_bits=config.get("input_bitwidth", 1),
        )
        layer_q_output_bits = [out_bits for _, out_bits, _ in layer_q_dims]

        print("\n[Layer-Q] Step B: generating cascade layer LUT files...")
        module_list_to_verilog_module(
            lut_model.module_list,
            "cascadelut",
            log_dir,
            add_registers=options_cfg["add_registers"],
            cascade_config=cascade_config,
        )

        print("\n[Layer-Q] Step C: generating unified Layer-Q cascadelut.v...")
        all_used_chunks = sorted(
            set(c for layer in lut_model.cascade_chunks for c in layer)
        )
        num_chunks = max(all_used_chunks) + 1 if all_used_chunks else 1

        layer_output_bits = []
        for m in lut_model.module_list:
            if isinstance(m, SparseLinearNeq):
                _, obw = m.output_quant.get_scale_factor_bits(m.cuda)
                layer_output_bits.append(int(m.out_features * int(obw)))

        _, final_obw = (
            lut_model.module_list[-1]
            .output_quant
            .get_scale_factor_bits(lut_model.is_cuda)
        )
        final_out_bits = int(lut_model.output_length * int(final_obw))

        top_verilog = generate_cascade_top_v2(
            top_name = "cascadelut",
            features_per_cycle = model_cfg["features_per_cycle"],
            num_chunks = num_chunks,
            cascade_chunks = lut_model.cascade_chunks,
            layer_output_bits = layer_output_bits,
            hw_bandwidth = hw_bandwidth,
            final_out_bits = final_out_bits,
            layer_q_output_bits = layer_q_output_bits,
        )
        with open(f"{log_dir}/cascadelut.v", "w") as fh:
            fh.write(top_verilog)
        print(f"   Written: {log_dir}/cascadelut.v  (overwrites Step B version)")
    else:
        print("\nGenerating cascade layers...")
        module_list_to_verilog_module(
            lut_model.module_list,
            "cascadelut",
            log_dir,
            add_registers=options_cfg["add_registers"],
            cascade_config=cascade_config,
            hw_bandwidth=hw_bandwidth,
        )
    print(f"\n   Top-level entity: {log_dir}/cascadelut.v")

    print("\n" + "=" * 70)
    print("6. Verilog simulation (PyVerilator)")
    print("=" * 70)
    if use_layer_q and layer_q_truth_tables is not None:
        verilog_inference_layer_q(
            lut_model,
            verilog_dir=log_dir,
            top_module_filename="cascadelut.v",
            hw_bandwidth=hw_bandwidth,
            num_chunks=num_chunks,
            feature_min_arr=feature_min_arr,
            feature_max_arr=feature_max_arr,
            layer_q_truth_tables=layer_q_truth_tables,
            logfile=None,
        )
    else:
        lut_model.verilog_inference(
            log_dir,
            "cascadelut.v",
            logfile=None,
            add_registers=options_cfg["add_registers"],
            hw_bandwidth=hw_bandwidth,
        )
    verilog_accuracy = test(lut_model, test_loader, cuda=options_cfg["cuda"])
    print(f"   Verilog-based accuracy: {verilog_accuracy:.6f}")

    fpga_part = "xc7z045ffg900-2"
    clk_period_ns = 5.0
    print("\n" + "=" * 70)
    print(f"7. In-context synthesis  (part={fpga_part}, clk={clk_period_ns} ns)")
    print("=" * 70)
    ret = synthesize_and_get_resource_counts(
        log_dir, "cascadelut",
        fpga_part=fpga_part,
        clk_period_ns=clk_period_ns,
        post_synthesis=1,
        ooc=False,
    )
    print(f"   Max f: {ret}")
    # for complete results use the full vivadopower method
    fast_power = get_power_analysis_fast(log_dir, "cascadelut")
    print(f"   Fast power: {fast_power}")