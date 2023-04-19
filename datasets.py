import os

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

SC = StandardScaler()


def get_dataset(root_dir, split_ratio):
    if "spect_heart" in root_dir:
        return get_spect_heart_dataset(root_dir, split_ratio)
    elif "breast_cancer" in root_dir:
        return get_breast_cancer_dataset(root_dir, split_ratio)
    elif "connectionist_bench" in root_dir:
        return get_connectionist_bench(root_dir, split_ratio)
    elif "somerville" in root_dir:
        return get_somerville_dataset(root_dir, split_ratio)
    elif "divorce" in root_dir:
        return get_divorce_predictor_dataset(root_dir, split_ratio)
    elif "ionosphere" in root_dir:
        return get_ionosphere_dataset(root_dir, split_ratio)
    elif "gun_point" in root_dir:
        return get_gun_point_dataset(root_dir, split_ratio)
    elif "coffee" in root_dir:
        return get_coffee_dataset(root_dir, split_ratio)
    elif "pima" in root_dir:
        return get_pima_dataset(root_dir, split_ratio)
    raise f"Dont have {root_dir} dataset"


def get_spect_heart_dataset(root_dir, split_ratio):
    train_data_path = os.path.join(root_dir, "SPECT.test")
    test_data_path = os.path.join(root_dir, "SPECT.train")

    # data = np.concatenate([np.loadtxt(train_data_path, delimiter=","),
    #                        np.loadtxt(test_data_path, delimiter=",")])
    # np.random.shuffle(data)

    # train_len = int((len(data)*split_ratio))+1
    # train_data = data[:train_len]
    # test_data = data[train_len:]

    train_data = np.loadtxt(train_data_path, delimiter=",")
    test_data = np.loadtxt(test_data_path, delimiter=",") 
    return torch.from_numpy(train_data).float(), torch.from_numpy(test_data).float()


def get_breast_cancer_dataset(root_dir, split_ratio):
    data_path = os.path.join(root_dir, "dataR2.csv")
    data = np.genfromtxt(data_path, delimiter=",")[1:, :]
    data[:, [9, 0]] = data[:, [0, 9]]

    data[:, 1:] = SC.fit_transform(data[:, 1:])
    np.random.seed(42)
    np.random.shuffle(data)
    train_len = int((len(data)*split_ratio))
    train_data = data[:train_len]
    test_data = data[train_len:]
    return torch.from_numpy(train_data).float(), torch.from_numpy(test_data).float()


def get_connectionist_bench(root_dir, split_ratio):
    data_path = os.path.join(root_dir, "sonar.all-data")
    data = np.genfromtxt(data_path, delimiter=",")
    data[:, [60, 0]] = data[:, [0, 60]]

    data[:, 1:] = SC.fit_transform(data[:, 1:])
    np.random.seed(42)
    np.random.shuffle(data)
    train_len = int((len(data)*split_ratio))
    train_data = data[:train_len]
    test_data = data[train_len:]
    return torch.from_numpy(train_data).float(), torch.from_numpy(test_data).float()


def get_somerville_dataset(root_dir, split_ratio):
    data_path = os.path.join(root_dir, "data")
    data = np.genfromtxt(data_path, delimiter=",")

    data[:, 1:] = SC.fit_transform(data[:, 1:])
    np.random.seed(42)
    np.random.shuffle(data)
    train_len = int((len(data)*split_ratio))
    train_data = data[:train_len]
    test_data = data[train_len:]
    return torch.from_numpy(train_data).float(), torch.from_numpy(test_data).float()


def get_divorce_predictor_dataset(root_dir, split_ratio):
    data_path = os.path.join(root_dir, "divorce.csv")
    data = np.genfromtxt(data_path, delimiter=";")[1:, :]
    data[:, [54, 0]] = data[:, [0, 54]]

    data[:, 1:] = SC.fit_transform(data[:, 1:])
    np.random.seed(42)
    np.random.shuffle(data)
    train_len = int((len(data)*split_ratio))
    train_data = data[:train_len]
    test_data = data[train_len:]
    return torch.from_numpy(train_data).float(), torch.from_numpy(test_data).float()


def get_ionosphere_dataset(root_dir, split_ratio):
    data_path = os.path.join(root_dir, "ionosphere.data")
    data = np.genfromtxt(data_path, delimiter=",")
    data[:, [34, 0]] = data[:, [0, 34]]

    data[:, 1:] = SC.fit_transform(data[:, 1:])
    np.random.seed(42)
    np.random.shuffle(data)
    train_len = int((len(data)*split_ratio))
    train_data = data[:train_len]
    test_data = data[train_len:]
    return torch.from_numpy(train_data).float(), torch.from_numpy(test_data).float()


def get_pima_dataset(root_dir, split_ratio):
    data_path = os.path.join(root_dir, "diabetes.csv")
    data = np.genfromtxt(data_path, delimiter=",")[1:, :]
    data[:, [8, 0]] = data[:, [0, 8]]

    data[:, 1:] = SC.fit_transform(data[:, 1:])
    np.random.seed(42)
    np.random.shuffle(data)
    train_len = int((len(data)*split_ratio))
    train_data = data[:train_len]
    test_data = data[train_len:]
    return torch.from_numpy(train_data).float(), torch.from_numpy(test_data).float()


def get_gun_point_dataset(root_dir, split_ratio):
    train_data_path = os.path.join(root_dir, "GunPoint_TRAIN.txt")
    test_data_path = os.path.join(root_dir, "GunPoint_TEST.txt")
    
    train_data = np.genfromtxt(train_data_path, delimiter="  ")
    test_data = np.genfromtxt(test_data_path, delimiter="  ")

    train_data[:, 1:] = SC.fit_transform(train_data[:, 1:])
    test_data[:, 1:] = SC.fit_transform(test_data[:, 1:])
    return torch.from_numpy(train_data).float(), torch.from_numpy(test_data).float()


def get_coffee_dataset(root_dir, split_ratio):
    train_data_path = os.path.join(root_dir, "Coffee_TRAIN.txt")
    test_data_path = os.path.join(root_dir, "Coffee_TEST.txt")
    
    train_data = np.genfromtxt(train_data_path, delimiter="  ")
    test_data = np.genfromtxt(test_data_path, delimiter="  ")

    train_data[:, 1:] = SC.fit_transform(train_data[:, 1:])
    test_data[:, 1:] = SC.fit_transform(test_data[:, 1:])
    return torch.from_numpy(train_data).float(), torch.from_numpy(test_data).float()