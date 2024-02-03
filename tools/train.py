import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from time import time
from dataclasses import dataclass
from tools.util import dict_reduce
from typing import Dict, List
import matplotlib.pyplot as plt


@dataclass
class HyperParams:
    epochs: int = 10
    lr_init: float = 0.01
    lr_decay: float = 0.9
    weight_decay: float = 0  # 0.0001 is a good starting value


class Metrics:
    def __init__(self):
        self.train = []
        self.test = []

    def push(self, model, train_loader: DataLoader, test_loader: DataLoader):
        self.train.append(evaluate_model_metrics(model, train_loader))
        self.test.append(evaluate_model_metrics(model, test_loader))
        print(
            f"Train {self.train[-1]['Accuracy']*100 :.2f}% | Test {self.test[-1]['Accuracy']*100:.2f}%"
        )


def get_correct_count(predictions, labels):
    return torch.sum((torch.argmax(predictions, dim=1) == labels).type(torch.int32))


def decayed_lr(hyper_params: HyperParams, epoch_i: int):
    return hyper_params.lr_init * hyper_params.lr_decay**epoch_i


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    for g in optimizer.param_groups:
        g["lr"] = lr


def evaluate_model_metrics(model: torch.nn.Module, loader: DataLoader):
    model.eval()  # <-- Switch model to 'eval' mode to notify layers like Dropout and BatchNorm that we don't plan to train the model
    with torch.no_grad():  # <-- Turns off Pytorch gradient calculations while evaluating the model to reduce memory and cpu use
        correct = 0
        for specs, _srs, labels in loader:
            predictions = model(specs)
            correct += get_correct_count(predictions, labels)
    return {"Accuracy": correct / len(loader.dataset)}


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn,
    optimizer: torch.optim.Optimizer,
):
    model.train()  # <-- Switch model to 'train' mode for layers like Dropout/BatchNorm
    for specs, _srs, labels in loader:
        # images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(specs)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()


def plot_metrics(metrics: Metrics):
    train_metrics = dict_reduce(metrics.train)
    test_metrics = dict_reduce(metrics.test)
    fig, axs = plt.subplots(
        len(train_metrics), 1, sharex=True, squeeze=False
    )  # figsize=(8, 6)
    i = 0
    for key in train_metrics:
        axs[i, 0].plot(train_metrics[key], label=f"Train")
        axs[i, 0].plot(test_metrics[key], label=f"Test")
        axs[i, 0].set_xlabel("Epoch")
        axs[i, 0].set_ylabel(key)
        axs[i, 0].legend()
        i += 1


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    hyper_params: HyperParams,
):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=hyper_params.weight_decay
    )
    device = "cpu"

    metrics = Metrics()
    print("----- Initialized model performance:")
    metrics.push(model, train_loader, test_loader)
    print()

    print("----- Trained Model Performance:")
    for epoch_i in range(hyper_params.epochs):
        start = time()
        lr = decayed_lr(hyper_params, epoch_i)
        set_lr(optimizer, lr)
        print(f"Epoch: {epoch_i + 1} | lr: {lr:.3}")
        train_epoch(model, train_loader, loss_fn, optimizer)
        metrics.push(model, train_loader, test_loader)
        print(f"Time: {time() - start:.1f}s\n")

    plot_metrics(metrics)
