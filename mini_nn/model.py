from typing import Dict, Iterable, List

from .layers import Layer
from .losses import Loss
from .optimizers import Optimizer
from .tensor import Parameter, Vector, mean


class Sequential:
    def __init__(self, layers: Iterable[Layer]):
        self.layers = list(layers)

    def forward(self, inputs: Vector) -> Vector:
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad_output: Vector) -> Vector:
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self) -> List[Parameter]:
        result: List[Parameter] = []
        for layer in self.layers:
            result.extend(layer.parameters())
        return result

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def fit(
        self,
        dataset,
        loss: Loss,
        optimizer: Optimizer,
        *,
        epochs: int = 100,
        batch_size: int = 1,
        shuffle: bool = True,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        history = []

        for epoch in range(1, epochs + 1):
            epoch_dataset = dataset.shuffle() if shuffle else dataset
            batch_losses = []

            for batch in epoch_dataset.minibatches(batch_size):
                self.zero_grad()
                current_losses = []

                for inputs, target in batch:
                    predicted = self.forward(inputs)
                    current_losses.append(loss.forward(predicted, target))
                    self.backward(loss.backward(predicted, target))

                optimizer.step(self.parameters(), grad_scale=1.0 / len(batch))
                batch_losses.append(mean(current_losses))

            metrics = {"epoch": float(epoch), "loss": mean(batch_losses)}
            history.append(metrics)

            if verbose and (epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0):
                print(f"epoch {epoch:4d}/{epochs}: loss={metrics['loss']:.6f}")

        return history

    def predict(self, inputs: Vector) -> Vector:
        return self.forward(inputs)

    def evaluate(self, dataset, loss: Loss) -> float:
        return mean(loss.forward(self.forward(inputs), target) for inputs, target in dataset)
