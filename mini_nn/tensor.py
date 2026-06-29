import math
import random
from typing import Callable, Iterable, List, Sequence, Union

NumberTree = Union[float, List["NumberTree"]]
Vector = List[float]
Matrix = List[List[float]]


class Parameter:
    def __init__(self, data: NumberTree):
        self.data = data
        self.grad = zeros_like(data)

    def zero_grad(self) -> None:
        self.grad = zeros_like(self.data)


def zeros_like(value: NumberTree) -> NumberTree:
    if isinstance(value, list):
        return [zeros_like(item) for item in value]
    return 0.0


def apply_in_place(
    data: NumberTree,
    grad: NumberTree,
    update: Callable[[float, float], float],
) -> None:
    if isinstance(data, list) and isinstance(grad, list):
        for item, item_grad in zip(data, grad):
            apply_in_place(item, item_grad, update)
        return

    raise TypeError("Top-level parameter data must be a list")


def update_nested(
    data: NumberTree,
    grad: NumberTree,
    update: Callable[[float, float], float],
) -> NumberTree:
    if isinstance(data, list) and isinstance(grad, list):
        for index, (item, item_grad) in enumerate(zip(data, grad)):
            data[index] = update_nested(item, item_grad, update)
        return data
    return update(float(data), float(grad))


def map_nested(value: NumberTree, func: Callable[[float], float]) -> NumberTree:
    if isinstance(value, list):
        return [map_nested(item, func) for item in value]
    return func(value)


def add_scaled(data: NumberTree, grad: NumberTree, scale: float) -> NumberTree:
    return update_nested(data, grad, lambda value, delta: value + scale * delta)


def clip(value: float, limit: float) -> float:
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def clipped_grad(grad: NumberTree, limit: float) -> NumberTree:
    return map_nested(grad, lambda value: clip(value, limit))


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def mat_vec(matrix: Matrix, vector: Vector) -> Vector:
    return [dot(row, vector) for row in matrix]


def add_vectors(left: Sequence[float], right: Sequence[float]) -> Vector:
    return [a + b for a, b in zip(left, right)]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def random_matrix(rows: int, cols: int, *, rng: random.Random) -> Matrix:
    limit = math.sqrt(6.0 / (rows + cols))
    return [[rng.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]


def zeros_vector(size: int) -> Vector:
    return [0.0 for _ in range(size)]
