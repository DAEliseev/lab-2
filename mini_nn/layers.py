import math
import random
from typing import List

from .tensor import Parameter, Vector, add_vectors, mat_vec, random_matrix, zeros_vector


class Layer:
    def forward(self, inputs: Vector) -> Vector:
        raise NotImplementedError

    def backward(self, grad_output: Vector) -> Vector:
        raise NotImplementedError

    def parameters(self) -> List[Parameter]:
        return []


class Dense(Layer):
    def __init__(self, input_size: int, output_size: int, *, seed: int | None = None):
        rng = random.Random(seed)
        self.weights = Parameter(random_matrix(output_size, input_size, rng=rng))
        self.biases = Parameter(zeros_vector(output_size))
        self.last_input: Vector = []

    def forward(self, inputs: Vector) -> Vector:
        self.last_input = inputs[:]
        return add_vectors(mat_vec(self.weights.data, inputs), self.biases.data)

    def backward(self, grad_output: Vector) -> Vector:
        grad_input = [0.0 for _ in self.last_input]

        for out_index, grad_value in enumerate(grad_output):
            self.biases.grad[out_index] += grad_value
            for in_index, input_value in enumerate(self.last_input):
                self.weights.grad[out_index][in_index] += grad_value * input_value
                grad_input[in_index] += self.weights.data[out_index][in_index] * grad_value

        return grad_input

    def parameters(self) -> List[Parameter]:
        return [self.weights, self.biases]


class ReLU(Layer):
    def __init__(self):
        self.last_input: Vector = []

    def forward(self, inputs: Vector) -> Vector:
        self.last_input = inputs[:]
        return [max(0.0, value) for value in inputs]

    def backward(self, grad_output: Vector) -> Vector:
        return [
            grad if input_value > 0.0 else 0.0
            for grad, input_value in zip(grad_output, self.last_input)
        ]


class Sigmoid(Layer):
    def __init__(self):
        self.last_output: Vector = []

    def forward(self, inputs: Vector) -> Vector:
        self.last_output = [1.0 / (1.0 + math.exp(-value)) for value in inputs]
        return self.last_output[:]

    def backward(self, grad_output: Vector) -> Vector:
        return [
            grad * output * (1.0 - output)
            for grad, output in zip(grad_output, self.last_output)
        ]


class Tanh(Layer):
    def __init__(self):
        self.last_output: Vector = []

    def forward(self, inputs: Vector) -> Vector:
        self.last_output = [math.tanh(value) for value in inputs]
        return self.last_output[:]

    def backward(self, grad_output: Vector) -> Vector:
        return [
            grad * (1.0 - output * output)
            for grad, output in zip(grad_output, self.last_output)
        ]


class Softmax(Layer):
    def __init__(self):
        self.last_output: Vector = []

    def forward(self, inputs: Vector) -> Vector:
        shifted = [value - max(inputs) for value in inputs]
        exps = [math.exp(value) for value in shifted]
        total = sum(exps)
        self.last_output = [value / total for value in exps]
        return self.last_output[:]

    def backward(self, grad_output: Vector) -> Vector:
        weighted_sum = sum(grad * prob for grad, prob in zip(grad_output, self.last_output))
        return [
            prob * (grad - weighted_sum)
            for grad, prob in zip(grad_output, self.last_output)
        ]
