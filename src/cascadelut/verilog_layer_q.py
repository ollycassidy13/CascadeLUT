#  This file is part of CascadeLUT.
#
#  CascadeLUT is a derivative work based on LogicNets,
#  which is licensed under the Apache License 2.0.

from __future__ import annotations
import math
from typing import List


def _clog2(n: int) -> int:
    return max(1, math.ceil(math.log2(max(n, 2))))


def compute_pipeline_depth(cascade_chunks: List[List[int]]) -> int:
    return 1 + sum(1 for chunks in cascade_chunks[1:] if chunks)


def compute_cycles_per_sample(
    num_chunks: int,
    features_per_cycle: int,
    hw_bandwidth: int,
    input_bitwidth: int = 8,  # For layer_q, always 8 (uint8 inputs)
) -> int:
    actual_bits = features_per_cycle * input_bitwidth
    load_cpc = math.ceil(actual_bits / hw_bandwidth)
    return num_chunks * load_cpc


def generate_cascade_top_v2(
    top_name: str,
    features_per_cycle: int,
    num_chunks: int,
    cascade_chunks: List[List[int]],
    layer_output_bits: List[int],
    hw_bandwidth: int,
    final_out_bits: int,
    layer_q_output_bits: List[int] = None,
) -> str:
    fpc_bits = features_per_cycle * 8
    load_cpc = math.ceil(fpc_bits / hw_bandwidth)
    raw_buf_bits = (load_cpc - 1) * hw_bandwidth
    total_cyc = num_chunks * load_cpc
    cnt_w = _clog2(total_cyc + 1)

    if layer_q_output_bits is None:  # default: 1-bit per feature
        layer_q_output_bits = [features_per_cycle] * num_chunks

    num_layers = len(layer_output_bits)
    num_hidden = num_layers - 1

    raw_buf_decls = "\n    ".join(
        f"reg [{raw_buf_bits-1}:0] raw_buf_{c};" for c in range(num_chunks)
    )
    raw_buf_rst = "\n            ".join(
        f"raw_buf_{c} <= {raw_buf_bits}'d0;" for c in range(num_chunks)
    )

    load_cases = ""
    for c in range(num_chunks):
        for k in range(load_cpc - 1):
            cycle = c * load_cpc + k
            lo = k * hw_bandwidth
            hi = lo + hw_bandwidth - 1
            load_cases += (
                f"                {cycle}: raw_buf_{c}[{hi}:{lo}] <= raw_input;\n"
            )
    load_cases += "                default: ;\n"

    lq_wires = "\n    ".join(
        f"wire [{layer_q_output_bits[c]-1}:0] lq_out_{c};" for c in range(num_chunks)
    )
    lq_insts = "\n    ".join(
        f"layer_q_{c} lq{c}_inst (.M0({{raw_input, raw_buf_{c}}}), .M1(lq_out_{c}));"
        for c in range(num_chunks)
    )

    we_decls = []
    we_rst_stmts = []
    we_captures = []

    layer_wire_decls = []
    layer_inst_lines = []

    for i in range(num_hidden):
        out_bits = layer_output_bits[i]
        chunks_here = cascade_chunks[i] if i < len(cascade_chunks) else []

        layer_wire_decls.append(f"wire [{out_bits-1}:0] layer_out_{i};")

        if not chunks_here:
            # Non-cascade: no register
            layer_inst_lines.append(
                f"layer{i} layer{i}_inst (.M0(layer_out_{i-1}), .M1(layer_out_{i}));"
            )
        else:
            # Cascade injection: WE register fires on the last load cycle of
            # the highest chunk index used by this layer.
            c_max = max(chunks_here)
            we_cycle = (c_max + 1) * load_cpc - 1   # last cycle of chunk c_max

            if i == 0:
                if len(chunks_here) == 1:
                    we_width = layer_q_output_bits[chunks_here[0]]
                    we_rhs = f"lq_out_{chunks_here[0]}"
                else:
                    we_width = sum(layer_q_output_bits[c] for c in chunks_here)
                    parts = ", ".join(f"lq_out_{c}" for c in reversed(chunks_here))
                    we_rhs = "{" + parts + "}"
            else:
                prev_bits = layer_output_bits[i - 1]
                if len(chunks_here) == 1:
                    we_width = prev_bits + layer_q_output_bits[chunks_here[0]]
                    # Python cat order: [prev, chunk] -> prev at low bits, chunk at high bits
                    # Verilog {A, B}: B is at low bits
                    we_rhs = (
                        "{" + f"lq_out_{chunks_here[0]}, layer_out_{i-1}" + "}"
                    )
                else:
                    chunk_bits = sum(layer_q_output_bits[c] for c in chunks_here)
                    we_width = prev_bits + chunk_bits
                    chunk_part = ", ".join(
                        f"lq_out_{c}" for c in reversed(chunks_here)
                    )
                    we_rhs = "{" + f"{chunk_part}, layer_out_{i-1}" + "}"

            reg_name = f"layer{i}_in_reg"
            we_decls.append(f"reg [{we_width-1}:0] {reg_name};")
            we_rst_stmts.append(f"{reg_name} <= {we_width}'d0;")
            we_captures.append((we_cycle, reg_name, we_rhs))

            layer_inst_lines.append(
                f"layer{i} layer{i}_inst (.M0({reg_name}), .M1(layer_out_{i}));"
            )

    final_idx = num_layers - 1
    layer_wire_decls.append(f"wire [{final_out_bits-1}:0] layer_out_{final_idx};")
    layer_inst_lines.append(
        f"layer{final_idx} layer{final_idx}_inst"
        f" (.M0(layer_out_{num_hidden-1}), .M1(layer_out_{final_idx}));"
    )

    sep4 = "\n    "
    sep12 = "\n            "

    we_decl_str = sep4.join(we_decls)       if we_decls      else ""
    we_rst_str = sep12.join(we_rst_stmts)  if we_rst_stmts  else ""

    we_body = ""
    for we_cycle, reg_name, rhs in we_captures:
        we_body += (
            f"            if (cycle_cnt == {we_cycle}) {reg_name} <= {rhs};\n\n"
        )

    return f"""\
module {top_name} (
    input  [{hw_bandwidth-1}:0]   raw_input,
    input  wire                   clk,
    input  wire                   rst,
    output [{final_out_bits-1}:0] output_data
);

    reg [{cnt_w-1}:0] cycle_cnt;

    {raw_buf_decls}

    {lq_wires}
    {lq_insts}

    {we_decl_str}

    {sep4.join(layer_wire_decls)}

    {sep4.join(layer_inst_lines)}

    assign output_data = layer_out_{final_idx};

    always @(posedge clk) begin
        if (rst) begin
            cycle_cnt    <= 0;
            {raw_buf_rst}
            {we_rst_str}
        end else begin

            cycle_cnt <= (cycle_cnt == {total_cyc-1}) ? {cnt_w}'d0
                                                      : cycle_cnt + 1;

            case (cycle_cnt)
{load_cases}            endcase

{we_body}
        end
    end

endmodule
"""