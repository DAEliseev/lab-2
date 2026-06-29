from mini_nn import BinaryCrossEntropy, Dataset, Dense, MomentumSGD, Sequential, Sigmoid, Tanh


def build_dataset() -> Dataset:
    return Dataset(
        [
            ([0.0, 0.0], [0.0]),
            ([0.0, 1.0], [1.0]),
            ([1.0, 0.0], [1.0]),
            ([1.0, 1.0], [0.0]),
        ]
    )


def run(verbose: bool = True) -> float:
    dataset = build_dataset()
    model = Sequential(
        [
            Dense(2, 4, seed=1),
            Tanh(),
            Dense(4, 1, seed=2),
            Sigmoid(),
        ]
    )

    history = model.fit(
        dataset,
        loss=BinaryCrossEntropy(),
        optimizer=MomentumSGD(lr=0.5, momentum=0.8),
        epochs=1200,
        batch_size=4,
        shuffle=True,
        verbose=False,
    )

    if verbose:
        print("XOR predictions:")
        for inputs, target in dataset:
            prediction = model.predict(inputs)[0]
            print(f"{inputs} -> {prediction:.3f} target={target[0]:.0f}")
        print(f"final loss: {history[-1]['loss']:.6f}")

    return history[-1]["loss"]


if __name__ == "__main__":
    run()
