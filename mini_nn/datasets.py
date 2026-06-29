import random
from typing import Callable, Iterable, Iterator, List, Sequence, Tuple

from .tensor import Vector

Example = Tuple[Vector, Vector]


class Dataset:
    def __init__(self, examples: Iterable[Example]):
        self.examples = [(list(inputs), list(target)) for inputs, target in examples]

    @classmethod
    def from_lists(cls, inputs: Sequence[Vector], targets: Sequence[Vector]) -> "Dataset":
        return cls(zip(inputs, targets))

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        return self.examples[index]

    def map(self, func: Callable[[Vector, Vector], Example]) -> "Dataset":
        return Dataset(func(inputs[:], target[:]) for inputs, target in self.examples)

    def shuffle(self, seed: int | None = None) -> "Dataset":
        rng = random.Random(seed)
        examples = self.examples[:]
        rng.shuffle(examples)
        return Dataset(examples)

    def minibatches(self, batch_size: int) -> Iterator[List[Example]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        for start in range(0, len(self.examples), batch_size):
            yield self.examples[start : start + batch_size]

    def normalize(self) -> "Dataset":
        if not self.examples:
            return Dataset([])

        feature_count = len(self.examples[0][0])
        means = []
        stds = []

        for feature_index in range(feature_count):
            values = [inputs[feature_index] for inputs, _ in self.examples]
            avg = sum(values) / len(values)
            variance = sum((value - avg) ** 2 for value in values) / len(values)
            means.append(avg)
            stds.append(variance ** 0.5 or 1.0)

        return self.map(
            lambda inputs, target: (
                [(value - means[index]) / stds[index] for index, value in enumerate(inputs)],
                target,
            )
        )


def train_test_split(
    dataset: Dataset,
    test_ratio: float = 0.2,
    *,
    seed: int | None = 42,
) -> Tuple[Dataset, Dataset]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1")

    shuffled = dataset.shuffle(seed).examples
    test_size = max(1, int(len(shuffled) * test_ratio))
    return Dataset(shuffled[test_size:]), Dataset(shuffled[:test_size])
