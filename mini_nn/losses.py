import math
from typing import List

Vector = List[float]


class Loss:
    def forward(self, predicted: Vector, target: Vector) -> float:
        raise NotImplementedError

    def backward(self, predicted: Vector, target: Vector) -> Vector:
        raise NotImplementedError


class MeanSquaredError(Loss):
    def forward(self, predicted: Vector, target: Vector) -> float:
        return sum((p - y) ** 2 for p, y in zip(predicted, target)) / len(target)

    def backward(self, predicted: Vector, target: Vector) -> Vector:
        size = len(target)
        return [2.0 * (p - y) / size for p, y in zip(predicted, target)]


class BinaryCrossEntropy(Loss):
    def __init__(self, epsilon: float = 1e-12):
        self.epsilon = epsilon

    def forward(self, predicted: Vector, target: Vector) -> float:
        losses = []
        for p, y in zip(predicted, target):
            p = min(max(p, self.epsilon), 1.0 - self.epsilon)
            losses.append(-(y * math.log(p) + (1.0 - y) * math.log(1.0 - p)))
        return sum(losses) / len(target)

    def backward(self, predicted: Vector, target: Vector) -> Vector:
        size = len(target)
        grads = []
        for p, y in zip(predicted, target):
            p = min(max(p, self.epsilon), 1.0 - self.epsilon)
            grads.append((p - y) / (p * (1.0 - p) * size))
        return grads


class CategoricalCrossEntropy(Loss):
    def __init__(self, epsilon: float = 1e-12):
        self.epsilon = epsilon

    def forward(self, predicted: Vector, target: Vector) -> float:
        total = 0.0
        for p, y in zip(predicted, target):
            p = min(max(p, self.epsilon), 1.0)
            total += -y * math.log(p)
        return total

    def backward(self, predicted: Vector, target: Vector) -> Vector:
        return [
            -y / min(max(p, self.epsilon), 1.0)
            for p, y in zip(predicted, target)
        ]
