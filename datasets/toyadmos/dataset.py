#  This file is part of CascadeLUT.
#
#  CascadeLUT is a derivative work based on LogicNets,
#  which is licensed under the Apache License 2.0.

import pickle

def load_dataset(dset_path):
    with open(f"{dset_path}/train.pkl", "rb") as f:
        train_dataset = pickle.load(f)
    with open(f"{dset_path}/test.pkl", "rb") as f:
        test_dataset = pickle.load(f)
    X_train = train_dataset[0].view(train_dataset[0].shape[0], -1).numpy()
    Y_train = train_dataset[1].numpy()
    X_test = test_dataset[0].view(test_dataset[0].shape[0], -1).numpy()
    Y_test = test_dataset[1].numpy()
    return X_train, Y_train, X_test, Y_test