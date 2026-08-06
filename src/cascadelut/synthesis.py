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
import subprocess
from shutil import which

#xcvu9p-flgb2104-2-i
# TODO: Add option to perform synthesis on a remote server
def synthesize_and_get_resource_counts(verilog_dir, top_name, fpga_part = "xcku3p-ffva676-1-e", clk_name="clk", clk_period_ns=5.0, post_synthesis = 0, ooc=False):
    # old part : "xczu3eg-sbva484-1-i"
    # ensure that the OH_MY_XILINX envvar is set
    if "OHMYXILINX" not in os.environ:
        raise Exception("The environment variable OHMYXILINX is not defined.")
    # ensure that vivado is in PATH: source $VIVADO_PATH/settings64.sh
    if which("vivado") is None:
        raise Exception("vivado is not in PATH, ensure settings64.sh is sourced.")
    abs_verilog_dir = os.path.abspath(verilog_dir)
    omx_path = os.environ["OHMYXILINX"]
    script = "vivadocompile.sh"
    # vivadocompile.sh <top-level-entity> <clock-name (optional)> <fpga-part (optional)>
    # oh-my-xilinx's 7th arg is now use_incontext (inverted from the old use_ooc), so negate ooc here
    call_omx = "zsh %s/%s %s %s %s %f %s %s %s" % (omx_path, script, top_name, clk_name, fpga_part, float(clk_period_ns), post_synthesis, abs_verilog_dir, str(int(not ooc)))
    call_omx = call_omx.split()
    proc = subprocess.Popen(call_omx, cwd=verilog_dir, stdout=subprocess.PIPE, env=os.environ)
    proc.communicate()

    vivado_proj_folder = "%s/results_%s" % (verilog_dir, top_name)
    res_counts_path = vivado_proj_folder + "/res.txt"

    with open(res_counts_path, 'r') as myfile:
        res_data = myfile.read().split("\n")
    ret = {}
    ret["vivado_proj_folder"] = vivado_proj_folder
    for res_line in res_data:
        res_fields = res_line.split("=")
        print(res_fields)
        try:
            ret[res_fields[0]] = float(res_fields[1])
        except ValueError:
            ret[res_fields[0]] = 0
        except IndexError:
            ret[res_fields[0]] = 0
    if ret["WNS"] == 0:
        ret["fmax_mhz"] = 0
    else:
        ret["fmax_mhz"] = 1000.0 / (float(clk_period_ns) - ret["WNS"])
    return ret

def get_power_analysis_fast(design_batch_dir, design_name, vivado_proj_dir=None):
    if "OHMYXILINX" not in os.environ:
        raise Exception("The environment variable OHMYXILINX is not defined.")
    if vivado_proj_dir is None:
        vivado_proj_dir = os.path.join(design_batch_dir, f"results_{design_name}")
    abs_proj_dir = os.path.abspath(vivado_proj_dir)
    omx_path = os.environ["OHMYXILINX"]
    call_cmd = ["zsh", f"{omx_path}/vivadopowerfast.sh", design_name, abs_proj_dir, "0"]
    proc = subprocess.Popen(call_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ)
    stdout, stderr = proc.communicate()
    output = stdout.decode('utf-8').strip()
    stderr_text = stderr.decode('utf-8').strip()
    if not output:
        raise Exception(
            "vivadopowerfast produced no output. "
            f"proj_dir={abs_proj_dir} stderr={stderr_text}"
        )
    # expected format: "design_name dynamic_power_mw total_power_mw"
    parts = output.split()
    if len(parts) >= 3:
        return {
            "design_name": parts[0],
            "dynamic_power_mw": float(parts[1].replace("mW", "")),
            "total_power_mw": float(parts[2].replace("mW", "")),
        }
    raise Exception(f"Failed to parse vivadopowerfast output: {output} stderr={stderr_text}")