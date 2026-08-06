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
from argparse import ArgumentParser
from functools import reduce

import numpy as np
import wandb
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from models import MnistCascadeModel, _parse_cascade_chunks
from cascadelut.quant import LearnablePerFeatureMultiBitActivation, ThresholdTemperatureScheduler

configs = {
    "hdr-5l-0.1": {
        "hidden_layers": [2160,360,2160,360,60],
        "support_layers": [0,1,0,1,1,1],
        "cascade_layers": [1, 0, 2, 0, 0],
        "input_bitwidth": 1,
        "hidden_bitwidth": 1,
        "output_bitwidth": 6,
        "support_bitwidth": 1,
        "input_fanin": 6,
        "hidden_fanin": 6,
        "output_fanin": 6, #this is not used
        "support_fanin": 6,
        "cascade_fanin": 8,  # fanin for cascade layers
        "width_n": 75,
        "weight_decay": 0.0002,
        "batch_size": 64,
        "epochs": 600,
        "warmup_epochs": 20,
        "learning_rate": 0.0016,
        "input_dropout": 0.08,
        "seed": 400,
        "checkpoint": None,
        "dense1_epochs": 16,
        "dense2_epochs": 37,
        "wd": 0.00012,
        "wd1": 0.03,
        "features_per_cycle": 112,
        "threshold_reg_strength": 0.000005,
        "threshold_reg_noise_std": 0.05,
        "start_temp": 1.8,
        "end_temp": 0.02,
        "label_smoothing": 0.05,
        "anneal_epochs": 100,
        "use_anneal": True
    },
}

# A dictionary, so we can set some defaults if necessary
model_config = {
    "hidden_layers": None,
    "support_layers": None,
    "cascade_layers": None,
    "input_bitwidth": None,
    "hidden_bitwidth": None,
    "output_bitwidth": None,
    "support_bitwidth": None,
    "input_fanin": None,
    "hidden_fanin": None,
    "output_fanin": None,
    "support_fanin": None,
    "cascade_fanin": None,
    "width_n": None,
    "features_per_cycle": None,
    "importance_order": None,
    "dense_forward": None,
    "imask": None
}

training_config = {
    "weight_decay": None,
    "batch_size": None,
    "epochs": None,
    "warmup_epochs": None,
    "learning_rate": None,
    "input_dropout": None,
    "seed": None,
    "dense1_epochs": None,
    "dense2_epochs": None,
    "wd": None,
    "wd1": None,

    "threshold_reg_strength": None,
    "threshold_reg_noise_std": None,
    "start_temp": None,
    "end_temp": None,
    "label_smoothing": None,
    "anneal_epochs": None,
    "use_anneal": None
}

other_options = {
    "cuda": None,
    "log_dir": None,
    "checkpoint": None,
    "device": 1
}

def reseed(seed, cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def train(model, train_cfg, options):
    train_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_loader = DataLoader(
        datasets.MNIST(
            "mnist_data",
            download=False,
            train=True,
            transform=train_transform,
        ),
        batch_size=train_cfg["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        datasets.MNIST(
            "mnist_data",
            download=False,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),  # first, convert image to PyTorch tensor
                    transforms.Normalize((0.1307,), (0.3081,)),  # normalize inputs
                ]
            ),
        ),
        batch_size=train_cfg["batch_size"],
        shuffle=False,
    )
    test_loader = DataLoader(
        datasets.MNIST(
            "mnist_data",
            download=False,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),  # first, convert image to PyTorch tensor
                    transforms.Normalize((0.1307,), (0.3081,)),  # normalize inputs
                ]
            ),
        ),
        batch_size=train_cfg["batch_size"],
        shuffle=False,
    )

    # Configure optimizer
    weight_decay = train_cfg["weight_decay"]
    decay_exclusions = [
        "bn",
        "bias",
        "learned_value",
        "log_temp",
    ]  # Make a list of parameters name fragments which will ignore weight decay TODO: make this list part of the train_cfg
    decay_params = []
    no_decay_params = []
    for pname, params in model.named_parameters():
        if params.requires_grad:
            if reduce(
                lambda a, b: a or b, map(lambda x: x in pname, decay_exclusions)
            ):  # check if the current label should be excluded from weight decay
                no_decay_params.append(params)
            else:
                decay_params.append(params)
    params = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = optim.AdamW(
        params,
        lr=train_cfg["learning_rate"],
        betas=(0.5, 0.999),
        weight_decay=weight_decay,
    )

    is_dense_training = (model_cfg.get('dense_forward') == True or
                        (hasattr(model, 'dense_forward') and model.dense_forward))

    # Configure scheduler
    steps = len(train_loader)
    if is_dense_training:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, steps * 100), T_mult=1
        )
    else:
        epochs_total = int(train_cfg["epochs"])
        steps_per_epoch = steps
        total_iters = max(1, epochs_total * steps_per_epoch)
        warmup_epochs = train_cfg["warmup_epochs"]
        warmup_iters = min(warmup_epochs * steps_per_epoch, max(0, total_iters - 1))
        if warmup_iters <= 0:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=max(1, total_iters // 5), T_mult=1, eta_min=1e-6
            )
        else:
            post_iters = max(1, total_iters - warmup_iters)
            lr_scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_iters),
                    optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(1, post_iters // 5), T_mult=1, eta_min=1e-6),
                ],
                milestones=[warmup_iters],
            )

    threshold_scheduler = None
    if any(hasattr(m, "thresholds") for m in model.modules()):
        if train_cfg["use_anneal"]:
            threshold_scheduler = ThresholdTemperatureScheduler(
                model,
                start_temp=train_cfg["start_temp"],
                end_temp=train_cfg["end_temp"],
                anneal_epochs=train_cfg["anneal_epochs"]
            )
        else:
            threshold_scheduler = ThresholdTemperatureScheduler(
                model,
                start_temp=train_cfg["start_temp"],
                end_temp=train_cfg["end_temp"]
            )

    # Configure criterion
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg["label_smoothing"])

    # Push the model to the GPU, if necessary
    if options["cuda"]:
        model.cuda()

    # Main training loop
    maxAcc = 0.0

    if is_dense_training:
        if model.is_cascade and not model.full_input_mode:
            num_epochs = train_cfg["dense2_epochs"]
        else:
            num_epochs = train_cfg["dense1_epochs"]
    else:
        num_epochs = train_cfg["epochs"]

    for epoch in range(0, num_epochs):
        if threshold_scheduler is not None and train_cfg["use_anneal"]:
            threshold_scheduler.step(epoch)
        # Train for this epoch
        model.train()
        accLoss = 0.0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if options["cuda"]:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            data = data.reshape(-1, 784)
            target = torch.nn.functional.one_hot(target, num_classes=10)

            output = model(data)

            loss = criterion(output, torch.max(target, 1)[1])

            if model_cfg.get('dense_forward') == True and hasattr(model.module_list[0], 'res2_dense'):
                loss = loss + regularizer(model)
            threshold_reg_strength = train_cfg.get("threshold_reg_strength") or 0.0
            if threshold_reg_strength > 0.0:
                loss = loss + compute_threshold_regularization(
                    model,
                    lambda_threshold=threshold_reg_strength,
                    add_noise=train_cfg.get("threshold_reg_noise_std", 0.0) > 0.0,
                    noise_std=train_cfg.get("threshold_reg_noise_std", 0.0),
                )
            pred = output.detach().max(1, keepdim=True)[1]
            target_label = torch.max(target.detach(), 1, keepdim=True)[1]
            curCorrect = pred.eq(target_label).long().sum()
            curAcc = 100.0 * curCorrect / len(data)
            correct += curCorrect
            accLoss += loss.detach() * len(data)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if threshold_scheduler is not None and not train_cfg["use_anneal"]:
                lr = optimizer.param_groups[0]["lr"]
                threshold_scheduler.step_with_lr(lr, train_cfg["learning_rate"])

        accLoss /= len(train_loader.dataset)
        accuracy = 100.0 * correct / len(train_loader.dataset)
        val_accuracy = test(model, val_loader, options["cuda"])
        test_accuracy = test(model, test_loader, options["cuda"])
        if model_cfg['dense_forward'] == False:
            modelSave = {
                "model_dict": model.state_dict(),
                "optim_dict": optimizer.state_dict(),
                "val_accuracy": val_accuracy,
                "test_accuracy": test_accuracy,
                "epoch": epoch,
            }
            # torch.save(modelSave, "test_" + options["log_dir"] + "/checkpoint.pth")
            if maxAcc < test_accuracy:
                torch.save(modelSave, "test_" + options["log_dir"] + "/best_accuracy.pth")
                maxAcc = test_accuracy
            wandb.log(
                {
                    "Train Acc (%)": accuracy.detach().cpu().numpy(),
                    "Train Loss(%)": accLoss.detach().cpu().numpy(),
                    "Test Acc (%)": test_accuracy,
                    "Valid Acc(%)": val_accuracy,
                }
            )
        else:
            print(f"Epoch: {epoch}/{num_epochs}\tTest Acc: {test_accuracy:.2f}")

def test(model, dataset_loader, cuda):
    model.eval()
    correct = 0
    accLoss = 0.0
    
    for batch_idx, (data, target) in enumerate(dataset_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data = data.reshape(-1, 784)
        target = torch.nn.functional.one_hot(target, num_classes=10)
        
        output = model(data)
        
        pred = output.detach().max(1, keepdim=True)[1]
        if cuda:
            pred = pred.cuda()
        target_label = torch.max(target.detach(), 1, keepdim=True)[1]
        curCorrect = pred.eq(target_label).long().sum()
        curAcc = 100.0 * curCorrect / len(data)
        correct += curCorrect
    accuracy = 100 * float(correct) / len(dataset_loader.dataset)
    return accuracy

def calculate_cascade_importance_order_frequency(model_full_input, model_cfg, is_cuda):
    # Rank features per chunk by frequency of dense-layer selection
    cascade_chunks = _parse_cascade_chunks(model_cfg["cascade_layers"])
    B = int(model_cfg["features_per_cycle"])
    L = int(model_cfg["input_length"])

    all_used_chunks = sorted(set(c for layer_chunks in cascade_chunks for c in layer_chunks))
    if not all_used_chunks:
        return torch.arange(L, dtype=torch.long)

    assigned = set()
    order_segments = []

    def _neuron_topk_indices_for_slice(module, row, fan_in, start_col, end_col):
        score = neuron_input_score(module, row)[start_col:end_col]
        k = min(fan_in, score.numel())
        return torch.topk(score, k).indices.tolist()

    for c in all_used_chunks:
        freq = np.zeros(L, dtype=np.int64)

        for layer_idx, layer_chunk_list in enumerate(cascade_chunks):
            if c not in layer_chunk_list:
                continue

            module = model_full_input.module_list[layer_idx]

            if layer_idx == 0:
                fan_in = int(model_cfg["input_fanin"])
                prev_size = 0
            else:
                fan_in = int(model_cfg.get("cascade_fanin", model_cfg["hidden_fanin"]))
                prev_size = int(model_cfg["hidden_layers"][layer_idx - 1])

            chunk_position = layer_chunk_list.index(c)
            chunk_features = L # full_input_mode
            start_col = prev_size + chunk_position * chunk_features
            end_col = start_col + chunk_features

            if not (hasattr(module, "res2_dense") and module.res2_dense is not None):
                continue
            in_dim = module.res2_dense.weight.shape[1]
            if end_col > in_dim:
                continue

            out_dim = module.res2_dense.weight.shape[0]
            for row in range(out_dim):
                local_idxs = _neuron_topk_indices_for_slice(module, row, fan_in, start_col, end_col)
                for li in local_idxs[:fan_in]:
                    gi = int(li)
                    if 0 <= gi < L:
                        freq[gi] += 1

        if assigned:
            freq[list(assigned)] = -1

        top = np.argsort(freq)[::-1]
        top = top[freq[top] >= 0]
        top = top[:B]

        order_segments.append(top.copy())
        assigned.update(top.tolist())

    remaining = np.array([i for i in range(L) if i not in assigned], dtype=np.int64)
    full_order = np.concatenate(order_segments + [remaining])
    return torch.as_tensor(full_order, dtype=torch.long)

@torch.no_grad()
def neuron_input_score(module, row: int) -> torch.Tensor:
    residual_score = module.res2_dense.weight[row].abs()
    if not (hasattr(module, 'fc1_dense') and module.fc1_dense is not None and
            hasattr(module, 'fc4') and module.fc4 is not None):
        return residual_score
    width_n = getattr(module, 'width_n', 1)
    start = row * width_n
    end = start + width_n
    w1 = module.fc1_dense.weight[start:end].abs()    # [width_n, in_features]
    w4 = module.fc4.weight[row].abs().unsqueeze(1)   # [width_n, 1]
    mlp_score = (w4 * w1).sum(dim=0)
    return residual_score + mlp_score

def GetInputMask(model, is_cuda):
    imask = torch.tensor([])
    if is_cuda:
        imask = imask.cuda()
    max_fan_in = max(
        model_cfg["hidden_fanin"],
        model_cfg["input_fanin"],
        model_cfg["output_fanin"],
        model_cfg["support_fanin"],
        model_cfg.get("cascade_fanin", model_cfg["hidden_fanin"])
    )
    for i in range(0, len(model_cfg["hidden_layers"]) + 1):
        module = model.module_list[i]
        if hasattr(module, 'dense_forward') and module.dense_forward:
            fan_in = module.fan_in
            for row in range(module.res2_dense.weight.size(0)):
                idx_final = torch.topk(neuron_input_score(module, row), fan_in).indices
                pad_len = max_fan_in - idx_final.numel()
                if pad_len > 0:
                    pad = torch.full((pad_len,), -1, device=idx_final.device)  # Use -1 padding to avoid biasing pixel 0
                    idx_final = torch.cat([idx_final, pad])
                imask_row = idx_final.unsqueeze(0)
                imask = torch.cat((imask, imask_row), dim=0)
    return imask.long()

def regularizer(model):
    R = 0
    for i in range(0,len(model_cfg["hidden_layers"])+1):
        module = model.module_list[i]
        if hasattr(module, 'dense_forward') and module.dense_forward:
            weight1 = module.res2_dense.weight
            weight2 = module.fc1_dense.weight
            res2_dense_sum = weight1.abs().sum(dim=-1)
            fc1_dense_sum = weight2.abs().sum(dim=-1)
            l1 = torch.cat((res2_dense_sum, fc1_dense_sum), dim=0)
            
            # Calculate R incrementally on the device
            R += train_cfg['wd'] * (torch.pow(2, train_cfg['wd1'] * l1) - 1).sum()
    return R

def compute_threshold_regularization(model, lambda_threshold=1e-3, add_noise=False, noise_std=0.05):
    reg_loss = 0.0
    for module in model.modules():
        if isinstance(module, LearnablePerFeatureMultiBitActivation):
            reg_loss += lambda_threshold * (module._sorted_thresholds() ** 2).sum()

            if add_noise and module.training:
                noise = torch.randn_like(module.base_threshold) * noise_std
                with torch.no_grad():
                    module.base_threshold.data.add_(noise)
                    module.base_threshold.data.clamp_(-2.0, 2.0)

    return reg_loss


if __name__ == "__main__":
    parser = ArgumentParser(description="CascadeLUT Example")
    parser.add_argument(
        "--arch",
        type=str,
        choices=configs.keys(),
        default="hdr-5l-0.1",
        metavar="",
        help="Specific the neural network model to use (default: %(default)s)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        metavar="",
        help="Weight decay (default: %(default)s)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        metavar="",
        help="Batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        metavar="",
        help="Number of epochs to train (default: %(default)s)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        metavar="",
        help="Initial learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--input_dropout",
        type=float,
        default=None,
        metavar="",
        help="Input dropout rate (default: %(default)s)",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=True,
        help="Train on a GPU (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="",
        help="Seed to use for RNG (default: %(default)s)",
    )
    parser.add_argument(
        "--input_bitwidth",
        type=int,
        default=None,
        metavar="",
        help="Bitwidth to use at the input (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden_bitwidth",
        type=int,
        default=None,
        metavar="",
        help="Bitwidth to use for activations in hidden layers (default: %(default)s)",
    )
    parser.add_argument(
        "--output_bitwidth",
        type=int,
        default=None,
        metavar="",
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
        "--input_fanin",
        type=int,
        default=None,
        metavar="",
        help="Fanin to use at the input (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden_fanin",
        type=int,
        default=None,
        metavar="",
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
        "--output_fanin",
        type=int,
        default=None,
        metavar="",
        help="Fanin to use at the output (default: %(default)s)",
    )
    parser.add_argument(
        "--width_n",
        type=int,
        default=None,
        metavar="",
        help="Width of sub-network(N) (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden_layers",
        nargs="+",
        type=int,
        default=None,
        metavar="",
        help="A list of hidden layer neuron sizes (default: %(default)s)",
    )
    # --cascade_layers is intentionally omitted from CLI
    parser.add_argument(
        "--log_dir",
        type=str,
        default="0",
        metavar="",
        help="A location to store the log output of the training run and the output model (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        metavar="",
        help="Retrain the model from a previous checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--device", 
        type=int, 
        default=1, 
        metavar="", 
        help="Device_id for GPU",
    )
    parser.add_argument(
        "--dense1_epochs",
        type=int,
        default=None,
        metavar="",
        help="Number of epochs to train the dense model (default: %(default)s)",
    )
    parser.add_argument(
        "--dense2_epochs",
        type=int,
        default=None,
        metavar="",
        help="Number of epochs to train the dense model (default: %(default)s)",
    )
    parser.add_argument(
        "--support_fanin",  
        type=int,
        default=None,
        metavar="",
        help="Fanin to use for the support layers (default: %(default)s)",
    )
    parser.add_argument(
        "--cascade_fanin",  
        type=int,
        default=None,
        metavar="",
        help="Fanin to use for cascade layers taking input chunks (default: %(default)s)",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=None,
        metavar="",
        help="Weight decay for group regularizer (default: %(default)s)",
    )
    parser.add_argument(
        "--wd1",
        type=float,
        default=None,
        metavar="",
        help="Weight decay for group regularizer (default: %(default)s)",
    )
    parser.add_argument(
        "--features_per_cycle",
        type=int,
        default=None,
        metavar="",
        help="Weight decay for group regularizer (default: %(default)s)",
    )
    parser.add_argument(
        "--threshold_reg_strength",
        type=float,
        default=None,
        metavar="",
        help="L2 regularization strength on learnable quantizer thresholds; 0 disables (default: %(default)s)",
    )
    parser.add_argument(
        "--threshold_reg_noise_std",
        type=float,
        default=None,
        metavar="",
        help="Std of Gaussian noise injected into base_threshold each step; 0 disables (default: %(default)s)",
    )

    args = parser.parse_args()
    defaults = configs[args.arch]
    options = vars(args)
    del options["arch"]
    config = {}
    for k in options.keys():
        config[k] = (
            options[k] if options[k] is not None else defaults[k]
        )  # Override defaults, if specified.
    
    if not os.path.exists("test_" + config["log_dir"]):
        os.makedirs("test_" + config["log_dir"])

    # Split up configuration options to be more understandable
    for k in defaults.keys():
        if k not in config:
            config[k] = defaults[k]

    model_cfg = {}
    for k in model_config.keys():
        if k in config:
            model_cfg[k] = config[k]
    train_cfg = {}
    for k in training_config.keys():
        if k in config:
            train_cfg[k] = config[k]
    options_cfg = {}
    for k in other_options.keys():
        if k in config:
            options_cfg[k] = config[k]
    model_cfg["cuda"] = options_cfg["cuda"]
    # Set random seeds
    random.seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])
    torch.manual_seed(train_cfg["seed"])
    os.environ["PYTHONHASHSEED"] = str(train_cfg["seed"])
    if options["cuda"]:
        torch.cuda.manual_seed(train_cfg["seed"])
        torch.cuda.manual_seed_all(train_cfg["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_device(options_cfg["device"])

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="CascadeLUT",
        # track hyperparameters and run metadata
        config={
            "hidden_layers": model_cfg["hidden_layers"],
            "support_layers": model_cfg["support_layers"],
            "cascade_layers": model_cfg["cascade_layers"],
            "input_bitwidth": model_cfg["input_bitwidth"],
            "hidden_bitwidth": model_cfg["hidden_bitwidth"],
            "output_bitwidth": model_cfg["output_bitwidth"],
            "support_bitwidth": model_cfg["support_bitwidth"],
            "input_fanin": model_cfg["input_fanin"],
            "hidden_fanin": model_cfg["hidden_fanin"],
            "output_fanin": model_cfg["output_fanin"],
            "support_fanin": model_cfg["support_fanin"],
            "cascade_fanin": model_cfg["cascade_fanin"],
            "width_n": model_cfg["width_n"],
            "weight_decay": train_cfg["weight_decay"],
            "batch_size": train_cfg["batch_size"],
            "epochs": train_cfg["epochs"],
            "learning_rate": train_cfg["learning_rate"],
            "input_dropout": train_cfg["input_dropout"],
            "seed": train_cfg["seed"],
            "dense1_epochs": train_cfg["dense1_epochs"],
            "dense2_epochs": train_cfg["dense2_epochs"],
            "wd": train_cfg["wd"],
            "wd1": train_cfg["wd1"],
            "dataset": "mnist",
            "features_per_cycle": model_cfg["features_per_cycle"],
            "threshold_reg_strength": train_cfg["threshold_reg_strength"],
            "threshold_reg_noise_std": train_cfg["threshold_reg_noise_std"],
            "warmup_epochs": train_cfg["warmup_epochs"],
            "start_temp": train_cfg["start_temp"],
            "end_temp": train_cfg["end_temp"],
            "label_smoothing": train_cfg["label_smoothing"],
            "anneal_epochs": train_cfg["anneal_epochs"],
            "use_anneal": train_cfg["use_anneal"],
        },
    )

    wandb.define_metric("Train Acc (%)", summary="max")
    wandb.define_metric("Test Acc (%)", summary="max")
    wandb.define_metric("Valid Acc(%)", summary="max")
    wandb.define_metric("Train Loss(%)", summary="min")

    assert model_cfg["input_bitwidth"] == model_cfg["hidden_bitwidth"], \
        "Late-input cascade currently requires input_bitwidth == hidden_bitwidth"
    assert len(model_cfg["support_layers"]) == len(model_cfg["hidden_layers"]) + 1, \
        "support_layers length must equal hidden_layers length + 1"
    assert len(model_cfg["cascade_layers"]) == len(model_cfg["hidden_layers"]), \
        "cascade_layers length must equal hidden_layers length"

    model_cfg["input_length"] = 784
    model_cfg["output_length"] = 10
    model_cfg['dense_forward'] = True
    model_cfg['imask'] = torch.tensor([]).cuda()

    model_cfg['full_input_mode'] = True
    model_cfg['imask'] = torch.tensor([]).cuda() if options_cfg["cuda"] else torch.tensor([])
    model_cfg['importance_order'] = torch.arange(model_cfg['input_length'], dtype=torch.long)

    reseed(train_cfg["seed"], options_cfg["cuda"])
    model_phase1 = MnistCascadeModel(model_cfg)
    if options_cfg["cuda"]:
        model_phase1.cuda()
    train(model_phase1, train_cfg, options_cfg)

    importance_order_tensor = calculate_cascade_importance_order_frequency(
        model_phase1, model_cfg, options_cfg["cuda"]
    )
    model_cfg['importance_order'] = importance_order_tensor
    model_cfg['full_input_mode'] = False
    torch.save(importance_order_tensor, "test_" + options_cfg["log_dir"] + "/importance_order.pth")

    B = model_cfg['features_per_cycle']
    cascade_chunks_vis = _parse_cascade_chunks(model_cfg['cascade_layers'])
    all_used_vis = sorted(set(c for lc in cascade_chunks_vis for c in lc))
    for c in all_used_vis:
        seg = importance_order_tensor[c * B : (c + 1) * B].tolist()
        print(f"  Chunk {c}: top-5 feature indices = {seg[:5]} ...")

    del model_phase1
    torch.cuda.empty_cache()
    
    model_cfg['dense_forward'] = True
    model_cfg['imask'] = torch.tensor([]).cuda() if options_cfg["cuda"] else torch.tensor([])

    model_cascade_dense = MnistCascadeModel(model_cfg)
    train(model_cascade_dense, train_cfg, options_cfg)

    imask_cascade = GetInputMask(model_cascade_dense, options_cfg["cuda"])
    print(f"Extracted cascade imask: shape={imask_cascade.shape}")
    torch.save(imask_cascade, "test_" + options_cfg["log_dir"] + "/imask_cascade.pth")

    phase2_state_dict = model_cascade_dense.state_dict()
    torch.save(phase2_state_dict, "test_" + options_cfg["log_dir"] + "/phase2_model.pth")

    model_cfg['dense_forward'] = False
    model_cfg['imask'] = imask_cascade.cuda() if options_cfg["cuda"] else imask_cascade

    model = MnistCascadeModel(model_cfg)

    os.remove("test_" + options_cfg["log_dir"] + "/phase2_model.pth")
    train(model, train_cfg, options_cfg)
    
    wandb.finish()
