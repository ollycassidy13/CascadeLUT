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
from collections import defaultdict

def generate_register_verilog(module_name="myreg", param_name="DataWidth", input_name="data_in", output_name="data_out"):
    register_template = """\
module {module_name} #(parameter {param_name}=16) (
    input [{param_name}-1:0] {input_name},
    input wire clk,
    input wire rst,
    output reg [{param_name}-1:0] {output_name}
    );
    always@(posedge clk) begin
    if(!rst)
        {output_name}<={input_name};
    else
        {output_name}<=0;
    end
endmodule\n
"""
    return register_template.format(    module_name=module_name,
                                        param_name=param_name,
                                        input_name=input_name,
                                        output_name=output_name)

def generate_logicnets_verilog(module_name: str, input_name: str, input_bits: int, output_name: str, output_bits: int, module_contents: str):
    logicnets_template = """\
module {module_name} (input [{input_bits_1:d}:0] {input_name}, input clk, input rst, output[{output_bits_1:d}:0] {output_name});
{module_contents}
endmodule\n"""
    return logicnets_template.format( module_name=module_name,
                                input_name=input_name,
                                input_bits_1=input_bits-1,
                                output_name=output_name,
                                output_bits_1=output_bits-1,
                                module_contents=module_contents)

def layer_connection_verilog(layer_string: str, input_string: str, input_bits: int, output_string: str, output_bits: int, output_wire=True, register=False):
    if register:
        layer_connection_template = """\
wire [{input_bits_1:d}:0] {input_string}w;
myreg #(.DataWidth({input_bits})) {layer_string}_reg (.data_in({input_string}), .clk(clk), .rst(rst), .data_out({input_string}w));\n"""
    else:
        layer_connection_template = """\
wire [{input_bits_1:d}:0] {input_string}w;
assign {input_string}w = {input_string};\n"""
    layer_connection_template += "wire [{output_bits_1:d}:0] {output_string};\n" if output_wire else ""
    layer_connection_template += "{layer_string} {layer_string}_inst (.M0({input_string}w), .M1({output_string}));\n"
    return layer_connection_template.format(    layer_string=layer_string,
                                                input_string=input_string,
                                                input_bits=input_bits,
                                                input_bits_1=input_bits-1,
                                                output_string=output_string,
                                                output_bits_1=output_bits-1)

def generate_lut_verilog(module_name, input_fanin_bits, output_bits, lut_string):
    lut_neuron_template = """\
module {module_name} ( input [{input_fanin_bits_1:d}:0] M0, output [{output_bits_1:d}:0] M1 );

	(*rom_style = "distributed" *) reg [{output_bits_1:d}:0] M1r;
	assign M1 = M1r;
	always @ (M0) begin
		case (M0)
{lut_string}
		endcase
	end
endmodule\n"""
    return lut_neuron_template.format(  module_name=module_name,
                                        input_fanin_bits_1=input_fanin_bits-1,
                                        output_bits_1=output_bits-1,
                                        lut_string=lut_string)

def generate_neuron_connection_verilog(input_indices, input_bitwidth):
    connection_string = ""
    for i in range(len(input_indices)):
        index = input_indices[i]
        offset = index*input_bitwidth
        for b in reversed(range(input_bitwidth)):
            connection_string += f"M0[{offset+b}]"
            if not (i == len(input_indices)-1 and b == 0):
                connection_string += ", "
    return connection_string

def cascade_gen_layer_verilog(self, module_prefix, directory, generate_bench=True):
    composite = self.input_quant
    _, output_bitwidth = self.output_quant.get_scale_factor_bits(self.cuda)
    output_bitwidth = int(output_bitwidth)
    total_input_bits = composite.get_total_input_bits()
    total_output_bits = self.out_features * output_bitwidth

    layer_contents = (
        f"module {module_prefix} "
        f"(input [{total_input_bits - 1}:0] M0, "
        f"output [{total_output_bits - 1}:0] M1);\n\n"
    )
    output_offset = 0

    for index in range(self.out_features):
        module_name = f"{module_prefix}_N{index}"
        indices, _, _, _ = self.neuron_truth_tables[index]
        neuron_verilog = self.gen_neuron_verilog(index, module_name)
        with open(f"{directory}/{module_name}.v", "w") as f:
            f.write(neuron_verilog)
        if generate_bench:
            neuron_bench = self.gen_neuron_bench(index, module_name)
            with open(f"{directory}/{module_name}.bench", "w") as f:
                f.write(neuron_bench)
        connection_string = generate_neuron_connection_verilog_composite(indices, composite)
        slot_bits = [composite.get_bits_for_feature(int(indices[j])) for j in range(len(indices))]
        wire_width = sum(slot_bits)
        wire_name = f"{module_name}_wire"
        connection_line = (
            f"wire [{wire_width - 1}:0] {wire_name} = {{{connection_string}}};\n"
        )
        inst_line = (
            f"{module_name} {module_name}_inst "
            f"(.M0({wire_name}), "
            f".M1(M1[{output_offset + output_bitwidth - 1}:{output_offset}]));\n\n"
        )
        layer_contents += connection_line + inst_line
        output_offset += output_bitwidth

    layer_contents += "endmodule"
    with open(f"{directory}/{module_prefix}.v", "w") as f:
        f.write(layer_contents)
    return total_input_bits, total_output_bits


def cascade_gen_neuron_verilog(self, index, module_name):
    (indices, input_perm_matrix, float_output_states, bin_output_states) = \
        self.neuron_truth_tables[index]
    composite = self.input_quant
    _, output_bitwidth = self.output_quant.get_scale_factor_bits(self.cuda)
    slot_bits = [composite.get_bits_for_feature(int(indices[j])) for j in range(len(indices))]
    cat_input_bitwidth = sum(slot_bits)
    lut_string = ""
    num_entries = input_perm_matrix.shape[0]
    for i in range(num_entries):
        entry_str = ""
        for idx in range(len(indices)):
            val = input_perm_matrix[i, idx]
            entry_str += composite.get_bin_str_for_slot(int(indices[idx]), val, is_cuda=self.cuda)
        res_str = self.output_quant.get_bin_str_from_int(bin_output_states[i], is_cuda=self.cuda)
        lut_string += (
            f"\t\t\t{int(cat_input_bitwidth)}'b{entry_str}: "
            f"M1r = {int(output_bitwidth)}'b{res_str};\n"
        )
    return generate_lut_verilog(module_name, int(cat_input_bitwidth), int(output_bitwidth), lut_string)


def generate_cascade_top_module(
    top_name: str,
    layer_output_bits: list,
    chunk_bits: int,             # features_per_cycle x input_bitwidth
    cascade_chunks: list = None,
    cascade_layers: list = None,
    hw_bandwidth: int = None,    # port width in bits; None -> chunk_bits (LOAD_CPC=1)
) -> str:
    num_layers = len(layer_output_bits)
    num_hidden = num_layers - 1

    eff_hw_bandwidth = hw_bandwidth if hw_bandwidth is not None else chunk_bits
    load_cpc = math.ceil(chunk_bits / eff_hw_bandwidth)

    if cascade_chunks is None:
        if load_cpc > 1:
            raise ValueError(
                "cascade_chunks must be explicitly set in cascade_config "
                "when hw_bandwidth < chunk_bits (LOAD_CPC > 1). "
                "cascade_layers fallback is not supported in buffered mode."
            )
        if cascade_layers is not None:
            parsed = []
            chunk_idx = 0
            assigned_chunks = []
            for cl in cascade_layers[:num_hidden]:
                if isinstance(cl, list):
                    refs = []
                    for ref in cl:
                        zero_idx = int(ref) - 1
                        if 0 <= zero_idx < len(assigned_chunks):
                            refs.append(assigned_chunks[zero_idx])
                    parsed.append(refs)
                else:
                    try:
                        cl_int = int(cl)
                    except (TypeError, ValueError):
                        cl_int = 0
                    if cl_int >= 1:
                        assigned_chunks.append(chunk_idx)
                        parsed.append([chunk_idx])
                        chunk_idx += 1
                    else:
                        parsed.append([])
            cascade_chunks = parsed
        else:
            cascade_chunks = [[0]] + [[] for _ in range(num_hidden - 1)]

    hw_bandwidth = eff_hw_bandwidth
    final_out_bits = layer_output_bits[-1]

    if not cascade_chunks[0]:
        raise ValueError("cascade_chunks[0] must contain at least one chunk index")

    all_used = sorted(set(c for layer in cascade_chunks for c in layer))
    num_chunks = max(all_used) + 1

    total_cyc    = num_chunks * load_cpc
    raw_buf_bits = (load_cpc - 1) * hw_bandwidth  # 0 when LOAD_CPC == 1

    cnt_w = max(1, (total_cyc - 1).bit_length())
    if total_cyc > 1:
        cnt_w = max(cnt_w, 2)

    # chunk_c_reg: only needed for layer 0 (no prev output to co-register) and for
    # non-max chunks whose WE fired earlier and must be held until c_max WE cycle.
    chunks_with_own_reg = set()
    layer_pipregs = {}

    for i in range(num_hidden):
        chunks_here = cascade_chunks[i] if i < len(cascade_chunks) else []
        if not chunks_here:
            continue
        if i == 0:
            for c in chunks_here:
                chunks_with_own_reg.add(c)
        else:
            c_max    = max(chunks_here)
            we_cyc   = (c_max + 1) * load_cpc - 1
            prev_bits = layer_output_bits[i - 1]
            in_width  = prev_bits + chunk_bits * len(chunks_here)

            # {highest_chunk, ..., lowest_chunk_reg, layer_out_{i-1}}:
            # MSBs = chunk features, LSBs = prev layer — matches torch.cat([prev, chunks])
            chunk_exprs = []
            for c in reversed(sorted(chunks_here)):
                if c == c_max:
                    live = f"{{input_chunk, raw_buf_{c}}}" if load_cpc > 1 else "input_chunk"
                    chunk_exprs.append(live)
                else:
                    chunks_with_own_reg.add(c)   # committed earlier, safe to read
                    chunk_exprs.append(f"chunk_{c}_reg")

            rhs = "{" + ", ".join(chunk_exprs) + f", layer_out_{i-1}" + "}"
            layer_pipregs[i] = dict(we_cycle=we_cyc, in_width=in_width, rhs=rhs)

    reg_decls = []
    rst_stmts = []
    for c in range(num_chunks):
        if load_cpc > 1:
            reg_decls.append(f"reg [{raw_buf_bits-1}:0] raw_buf_{c};")
            rst_stmts.append(f"raw_buf_{c} <= {raw_buf_bits}'d0;")
        if c in chunks_with_own_reg:
            reg_decls.append(f"reg [{chunk_bits-1}:0] chunk_{c}_reg;")
            rst_stmts.append(f"chunk_{c}_reg <= {chunk_bits}'d0;")
    for i in sorted(layer_pipregs):
        w = layer_pipregs[i]['in_width']
        reg_decls.append(f"reg [{w-1}:0] layer{i}_in_reg;")
        rst_stmts.append(f"layer{i}_in_reg <= {w}'d0;")
    reg_decls.append(f"reg [{cnt_w-1}:0] cycle_cnt;")
    rst_stmts.append("cycle_cnt <= 0;")

    cycle_actions = defaultdict(list)

    for c in range(num_chunks):
        if load_cpc > 1:
            for k in range(load_cpc - 1):
                cyc = c * load_cpc + k
                lo  = k * hw_bandwidth
                hi  = lo + hw_bandwidth - 1
                cycle_actions[cyc].append(f"raw_buf_{c}[{hi}:{lo}] <= input_chunk;")

    for c in chunks_with_own_reg:
        we_cyc = (c + 1) * load_cpc - 1
        live   = f"{{input_chunk, raw_buf_{c}}}" if load_cpc > 1 else "input_chunk"
        cycle_actions[we_cyc].append(f"chunk_{c}_reg <= {live};")

    for i, info in sorted(layer_pipregs.items()):
        cycle_actions[info['we_cycle']].append(
            f"layer{i}_in_reg <= {info['rhs']};"
        )

    case_lines = []
    for cyc in sorted(cycle_actions):
        acts = cycle_actions[cyc]
        if len(acts) == 1:
            case_lines.append(f"                {cyc}: {acts[0]}")
        else:
            case_lines.append(f"                {cyc}: begin")
            for a in acts:
                case_lines.append(f"                    {a}")
            case_lines.append(f"                end")
    case_lines.append("                default: ;")

    layer_wire_decls = []
    layer_inst_lines = []

    for i in range(num_hidden):
        out_bits    = layer_output_bits[i]
        chunks_here = cascade_chunks[i] if i < len(cascade_chunks) else []
        layer_wire_decls.append(f"wire [{out_bits-1}:0] layer_out_{i};")

        if i == 0:
            if len(chunks_here) == 1:
                m0 = f"chunk_{chunks_here[0]}_reg"
            else:
                parts = ", ".join(f"chunk_{c}_reg" for c in reversed(sorted(chunks_here)))
                m0 = "{" + parts + "}"
            layer_inst_lines.append(
                f"layer0 layer0_inst (.M0({m0}), .M1(layer_out_0));"
            )
        elif not chunks_here:
            layer_inst_lines.append(
                f"layer{i} layer{i}_inst (.M0(layer_out_{i-1}), .M1(layer_out_{i}));"
            )
        else:
            layer_inst_lines.append(
                f"layer{i} layer{i}_inst (.M0(layer{i}_in_reg), .M1(layer_out_{i}));"
            )

    final_idx = num_layers - 1
    layer_wire_decls.append(f"wire [{final_out_bits-1}:0] layer_out_{final_idx};")
    layer_inst_lines.append(
        f"layer{final_idx} layer{final_idx}_inst"
        f" (.M0(layer_out_{num_hidden-1}), .M1(layer_out_{final_idx}));"
    )

    wrap_expr = f"(cycle_cnt == {total_cyc-1}) ? {cnt_w}'d0 : cycle_cnt + 1"

    template = """\
module {top_name} (
    input  [{hw_bw_1}:0] input_chunk,
    input  wire clk,
    input  wire rst,
    output [{final_out_bits_1}:0] output_data
);
    {reg_decls}

    {layer_wire_decls}

    {layer_inst_lines}

    assign output_data = layer_out_{final_idx};

    always @(posedge clk) begin
        if (rst) begin
            {rst_stmts}
        end else begin
            cycle_cnt <= {wrap_expr};
            case (cycle_cnt)
{case_body}
            endcase
        end
    end

endmodule
"""
    return template.format(
        top_name         = top_name,
        hw_bw_1          = hw_bandwidth - 1,
        final_out_bits_1 = final_out_bits - 1,
        final_idx        = final_idx,
        reg_decls        = "\n    ".join(reg_decls),
        layer_wire_decls = "\n    ".join(layer_wire_decls),
        layer_inst_lines = "\n    ".join(layer_inst_lines),
        rst_stmts        = "\n            ".join(rst_stmts),
        wrap_expr        = wrap_expr,
        case_body        = "\n".join(case_lines),
    )
