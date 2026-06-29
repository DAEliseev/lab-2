from typing import Dict, Iterable

from .tensor import NumberTree, Parameter, clipped_grad, update_nested, zeros_like


class Optimizer:
    def step(self, parameters: Iterable[Parameter], grad_scale: float = 1.0) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self, parameters: Iterable[Parameter], grad_scale: float = 1.0) -> None:
        for parameter in parameters:
            update_nested(
                parameter.data,
                parameter.grad,
                lambda value, grad: value - self.lr * grad * grad_scale,
            )


class MomentumSGD(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity: Dict[int, NumberTree] = {}

    def step(self, parameters: Iterable[Parameter], grad_scale: float = 1.0) -> None:
        for parameter in parameters:
            key = id(parameter)
            if key not in self.velocity:
                self.velocity[key] = zeros_like(parameter.data)

            self.velocity[key] = update_velocity(
                self.velocity[key],
                parameter.grad,
                self.momentum,
                self.lr,
                grad_scale,
            )
            update_nested(parameter.data, self.velocity[key], lambda value, delta: value + delta)


class GradientClippingSGD(SGD):
    def __init__(self, lr: float = 0.01, clip_value: float = 1.0):
        super().__init__(lr=lr)
        self.clip_value = clip_value

    def step(self, parameters: Iterable[Parameter], grad_scale: float = 1.0) -> None:
        for parameter in parameters:
            grad = clipped_grad(parameter.grad, self.clip_value)
            update_nested(
                parameter.data,
                grad,
                lambda value, clipped: value - self.lr * clipped * grad_scale,
            )


def update_velocity(
    velocity: NumberTree,
    grad: NumberTree,
    momentum: float,
    lr: float,
    grad_scale: float,
) -> NumberTree:
    if isinstance(velocity, list) and isinstance(grad, list):
        for index, (velocity_item, grad_item) in enumerate(zip(velocity, grad)):
            velocity[index] = update_velocity(velocity_item, grad_item, momentum, lr, grad_scale)
        return velocity
    return momentum * float(velocity) - lr * float(grad) * grad_scale
