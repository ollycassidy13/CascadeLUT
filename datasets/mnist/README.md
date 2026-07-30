## CascadeLUT on the MNIST dataset

This folder provides the code and resources to create CascadeLUT models on the MNIST dataset.

### Demo Notebook
For a quick introduction to the CascadeLUT flow, check out `cascade_demo.ipynb`.

This notebook:

* Walks you through the 3 stages of training
* Visualizes the learned feature salience, highlighting which pixels are transmitted and when.
* Generates the truth tables
* Runs a software simulation on the truth tables to validate accuracy
* Generates Verilog files

### Start Guide

To train a full model, run a hardware simulation with Verilator and perform synthesis and implementation through Vivado, follow these steps:

1. Train the Model
```
python train.py --arch hdr-5l-0.1 --log_dir cascade_demo --cuda --device 1
```

2. Convert to Verilog, Simulate and Evaluate
This script:
* Loads the trained checkpoint
* Verifies test accuracy
* Generates truth tables
* Runs both software simulation and hardware simulation using Verilator
* Compiles Verilog files for FPGA inference

```
python neq2lut.py --arch hdr-5l-0.1 \
                  --checkpoint ./test_cascade_demo/best_accuracy.pth \
                  --log-dir ./test_cascade_demo/verilog/ \
                  --add-registers \
                  --device 1 \
                  --imask ./test_cascade_demo/imask_cascade.pth \
                  --cuda
				  --hw-bandwidth 112
```

For input quantization:
* Adds a layer of L-LUTs which quantize the input from `int8`
* Verifies test accuracy with this layer
* Generates truth tables
* Runs both software simulation and hardware simulation using Verilator
* Compiles Verilog files for FPGA inference

```
python neq2lut.py --arch hdr-5l-0.1 \
                  --checkpoint ./test_cascade_demo/best_accuracy.pth \
                  --log-dir ./test_cascade_demo/verilog/ \
                  --add-registers \
                  --device 1 \
                  --imask ./test_cascade_demo/imask_cascade.pth \
                  --cuda
				  --hw-bandwidth 112
				  --use_layer_q
```

> NOTE: The parameters set for training here are not the parameters used in the paper.

## Citation
If CascadeLUT contributes to your research, please cite:

```bibtex
@inproceedings{cassidy2026cascadelut,
    author    = {Cassidy, Oliver and Andronic, Marta and Constantinides, George A.},
    title     = {{CascadeLUT: Information-Ordered Streaming Inference for Bandwidth-Constrained FPGAs}},
    booktitle = {2026 36th International Conference on Field-Programmable Logic and Applications (FPL)},
    publisher = {IEEE},
    year      = {2026}
}
```

CascadeLUT builds on NeuraLUT-Assemble; please also cite:

```bibtex
@inproceedings{andronic2025neuralut-assemble,
    author    = "Andronic, Marta and Constantinides, George A.",
    title     = "{NeuraLUT-Assemble: Hardware-Aware Assembling of Sub-Neural Networks for Efficient LUT Inference}",
    booktitle = "{2025 IEEE 33rd Annual International Symposium on Field-Programmable Custom Computing Machines (FCCM)}",
    pages     = "208-216",
    publisher = "IEEE",
    year      = 2025,
    note      = "doi: 10.1109/FCCM62733.2025.00077"
}
```