from mini_nn import Dataset, Dense, GradientClippingSGD, MeanSquaredError, Sequential, Tanh


def build_dataset() -> Dataset:
    examples = []
    for value in range(-20, 21):
        x = value / 10.0
        y = 0.5 * x * x - 0.25 * x + 0.1
        examples.append(([x], [y]))
    return Dataset(examples).normalize()


def run(verbose: bool = True) -> float:
    dataset = build_dataset()
    model = Sequential(
        [
            Dense(1, 8, seed=3),
            Tanh(),
            Dense(8, 1, seed=4),
        ]
    )

    history = model.fit(
        dataset,
        loss=MeanSquaredError(),
        optimizer=GradientClippingSGD(lr=0.05, clip_value=1.0),
        epochs=350,
        batch_size=8,
        shuffle=True,
        verbose=False,
    )

    if verbose:
        print("Regression predictions:")
        for inputs, target in [dataset[0], dataset[len(dataset) // 2], dataset[-1]]:
            prediction = model.predict(inputs)[0]
            print(f"x={inputs[0]:.3f} -> {prediction:.3f} target={target[0]:.3f}")
        print(f"final loss: {history[-1]['loss']:.6f}")

    return history[-1]["loss"]


if __name__ == "__main__":
    run()
