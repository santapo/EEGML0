import math
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall
from tqdm import tqdm

from datasets import get_dataset
from losses import EEGMLoss
from sgd import SGD
from utils import read_config


class TwoLayerNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.block = nn.Sequential(
            nn.Linear(self.num_features, 1),
            nn.Sigmoid()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -0.2, 0.2)
            nn.init.uniform_(m.bias, -0.2, 0.2)

    def forward(self, x):
        return self.block(x)


def get_regularizer(name, sigma=0.08):
    if name == "regularizer_1":
        return lambda x: 0.5*sigma**2*x*torch.exp(-0.5*x**2/sigma**2)
    elif name == "regularizer_2":
        return lambda x: 2*sigma**2*x/(x**2+sigma**2)**2
    elif name == "regularizer_3":
        return lambda x: (sigma*torch.sin(x/sigma)-x*math.cos(1/sigma))/x**2
    elif name == "regularizer_4":
        return lambda x: 4/sigma*x/(torch.exp(0.5*x**2/sigma)+torch.exp(-0.5*x**2/sigma))**2


def train(config):
    metrics = MetricCollection({
        "accuracy": Accuracy(task="binary"),
        "precision": Precision(task="binary"),
        "recall": Recall(task="binary"),
        "f1score": F1Score(task="binary"),
    })
    train_metrics = metrics.clone(prefix="train_")
    val_metrics = metrics.clone(prefix="val_")

    train_data, test_data = get_dataset(config["dataset_path"], config["split_ratio"])

    train_metrics_data = np.zeros((config["num_runs"], 5, config["num_epochs"])) # 4 metric + loss
    val_metrics_data = np.zeros((config["num_runs"], 5, config["num_epochs"])) # 4 metric + loss
    for j in tqdm(range(config["num_runs"])):
        model = TwoLayerNN(config["num_features"])
        model.to("cuda")
        optimizer = SGD(model.parameters(),
                        lr=config["learning_rate"],
                        weight_decay=config["weight_decay"],
                        regularizer=get_regularizer(config["regularizer"], config["sigma"]))
        loss_fn = EEGMLoss()

        train_metrics = metrics.clone(prefix=f"train_")
        val_metrics = metrics.clone(prefix=f"val_")

        for i in range(config["num_epochs"]):
            # Forward pass
            model.train()
            inputs, labels = train_data[:, 1:], train_data[:, :1]
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            outputs = model(inputs)

            train_metrics.update(outputs.cpu(), labels.cpu())
            calculated_train_metrics = train_metrics.compute()
            train_metrics.reset()

            loss = loss_fn(outputs, labels)
            # if loss.item() < 1e-3:
            #     break

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # eval
            model.eval()
            inputs, labels = test_data[:, 1:], test_data[:, :1]
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            outputs = model(inputs)

            val_metrics.update(outputs.cpu(), labels.cpu())
            calculated_val_metrics = val_metrics.compute()
            val_metrics.reset()
            val_loss = loss_fn(outputs, labels)

            train_metrics_data[j, :, i] = np.array(list(calculated_train_metrics.values())+[loss.item()])
            val_metrics_data[j, :, i] = np.array(list(calculated_val_metrics.values())+[val_loss.item()])
    return train_metrics_data, val_metrics_data


if __name__ == "__main__":
    # parse arguments by reading in a config
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = read_config(args.config)

    print(f"Start the training job with config: {config}")
    train_metrics_data, val_metrics_data = train(config)
    print("The training job is done!")
    print("=====================================")
    print("Best Accuracy score:")
    print("Training Accuracy: ", np.max(train_metrics_data[:, 0].mean(axis=0)))
    print("Validation Accuracy: ", np.max(val_metrics_data[:, 0].mean(axis=0)))