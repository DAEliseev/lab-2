from mini_nn import (
    CategoricalCrossEntropy,
    Dataset,
    Dense,
    MomentumSGD,
    Sequential,
    Softmax,
    Tanh,
    train_test_split,
)


def one_hot(index: int, size: int = 3):
    return [1.0 if item == index else 0.0 for item in range(size)]


def argmax(values):
    return max(range(len(values)), key=lambda index: values[index])


def build_dataset() -> Dataset:
    rows = []
    class_centers = [
        ([5.0, 3.4, 1.4, 0.2], 0),
        ([5.9, 2.8, 4.3, 1.3], 1),
        ([6.5, 3.0, 5.5, 2.0], 2),
    ]
    offsets = [-0.20, -0.10, 0.00, 0.10, 0.20, 0.30]

    for center, class_index in class_centers:
        for offset in offsets:
            features = [
                center[0] + offset,
                center[1] - offset * 0.3,
                center[2] + offset * 0.7,
                center[3] + offset * 0.2,
            ]
            rows.append((features, one_hot(class_index)))

    return Dataset(rows).normalize()


def run(verbose: bool = True) -> float:
    train_data, test_data = train_test_split(build_dataset(), test_ratio=0.25, seed=7)
    model = Sequential(
        [
            Dense(4, 6, seed=5),
            Tanh(),
            Dense(6, 3, seed=6),
            Softmax(),
        ]
    )

    model.fit(
        train_data,
        loss=CategoricalCrossEntropy(),
        optimizer=MomentumSGD(lr=0.08, momentum=0.8),
        epochs=250,
        batch_size=4,
        shuffle=True,
        verbose=False,
    )

    correct = 0
    for inputs, target in test_data:
        predicted_class = argmax(model.predict(inputs))
        correct += int(predicted_class == argmax(target))

    accuracy = correct / len(test_data)

    if verbose:
        print("Iris-like classification:")
        print(f"test accuracy: {accuracy:.2%} ({correct}/{len(test_data)})")

    return accuracy


if __name__ == "__main__":
    run()
