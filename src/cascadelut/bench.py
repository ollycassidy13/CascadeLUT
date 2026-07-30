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

def generate_lut_bench(input_fanin_bits, output_bits, lut_string):
    lut_neuron_template = """\
{input_string}\
{output_string}\
{lut_string}"""
    input_string = ""
    for i in range(input_fanin_bits):
        input_string += f"INPUT(M0[{i}])\n"
    output_string = ""
    for i in range(output_bits):
        output_string += f"OUTPUT(M1[{i}])\n"
    return lut_neuron_template.format(  input_string=input_string,
                                        output_string=output_string,
                                        lut_string=lut_string)

def generate_lut_input_string(input_fanin_bits):
    lut_input_string = ""
    for i in range(input_fanin_bits):
        if i == 0:
            lut_input_string += f"( M0[{i}]"
        elif i == input_fanin_bits-1:
            lut_input_string += f", M0[{i}] )\n"
        else:
            lut_input_string += f", M0[{i}]"
    return lut_input_string

def sort_to_bench(input_state_space_bin_str, bin_output_states):
    sorted_bin_output_states = bin_output_states.tolist()
    input_state_space_flat_int = list(map(lambda l: int(reduce(lambda a,b: a+b, l),2), input_state_space_bin_str))
    zipped_io_states = list(zip(input_state_space_flat_int, sorted_bin_output_states))
    zipped_io_states.sort(key=lambda x: x[0], reverse=True)
    sorted_bin_output_states = list(map(lambda x: x[1], zipped_io_states))
    return sorted_bin_output_states

def cascade_gen_neuron_bench(self, index, module_name):
    (indices, input_perm_matrix, float_output_states, bin_output_states) = \
        self.neuron_truth_tables[index]
    composite = self.input_quant
    _, output_bitwidth = self.output_quant.get_scale_factor_bits(self.cuda)
    slot_bits = [composite.get_bits_for_feature(int(indices[j])) for j in range(len(indices))]
    cat_input_bitwidth = sum(slot_bits)
    input_state_space_bin_str = []
    for row in input_perm_matrix:
        row_strs = []
        for idx in range(len(indices)):
            row_strs.append(
                composite.get_bin_str_for_slot(int(indices[idx]), row[idx], is_cuda=self.cuda)
            )
        input_state_space_bin_str.append(row_strs)
    sorted_bin_output_states = sort_to_bench(input_state_space_bin_str, bin_output_states)
    num_entries = input_perm_matrix.shape[0]
    lut_string = ""
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
        lut_hex_string = f"{int(output_bin_str, 2):0{int(num_entries / 4)}x} "
        lut_string += lut_hex_string
        lut_string += generate_lut_input_string(int(cat_input_bitwidth))
    return generate_lut_bench(int(cat_input_bitwidth), int(output_bitwidth), lut_string)
