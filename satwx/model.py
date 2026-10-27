"""ImageNet-pretrained ResNet-18 adapted to N-channel IR input, plus temperature scaling."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision


def build_resnet18(in_ch: int, n_classes: int, pretrained: bool = True) -> nn.Module:
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = torchvision.models.resnet18(weights=weights)
    old = net.conv1
    net.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():  # reuse RGB filters: average over colour, replicate over IR channels
        net.conv1.weight.copy_(old.weight.mean(1, keepdim=True).repeat(1, in_ch, 1, 1) * 3 / in_ch)
    net.fc = nn.Linear(net.fc.in_features, n_classes)
    return net


class Classifier:
    """Bundles network, input normalisation and calibration temperature."""

    def __init__(self, net, classes, channels, mean, std, temperature=1.0):
        self.net, self.classes, self.channels = net.eval(), list(classes), list(channels)
        self.mean = np.asarray(mean, np.float32).reshape(-1, 1, 1)
        self.std = np.asarray(std, np.float32).reshape(-1, 1, 1)
        self.temperature = float(temperature)

    def normalise(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, np.float32) - self.mean) / self.std

    @torch.no_grad()
    def logits(self, x: np.ndarray, batch: int = 512) -> np.ndarray:
        """x: (N, C, 64, 64) brightness temperatures in K."""
        out = []
        for i in range(0, len(x), batch):
            out.append(self.net(torch.from_numpy(self.normalise(x[i:i + batch]))).numpy())
        return np.concatenate(out)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.logits(x) / self.temperature)

    def save(self, path: str | Path, **extra):
        torch.save(dict(state=self.net.state_dict(), classes=self.classes, channels=self.channels,
                        mean=self.mean.ravel().tolist(), std=self.std.ravel().tolist(),
                        temperature=self.temperature, **extra), path)

    @classmethod
    def load(cls, path: str | Path) -> "Classifier":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = build_resnet18(len(ck["channels"]), len(ck["classes"]), pretrained=False)
        net.load_state_dict(ck["state"])
        return cls(net, ck["classes"], ck["channels"], ck["mean"], ck["std"], ck["temperature"])


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature minimising NLL on held-out logits (Guo et al. 2017)."""
    lg, yt = torch.from_numpy(logits).double(), torch.from_numpy(y).long()
    log_t = torch.zeros(1, dtype=torch.double, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(lg / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)
