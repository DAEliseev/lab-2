from examples.iris_like import run as run_iris_like
from examples.regression import run as run_regression
from examples.xor import run as run_xor


def main() -> None:
    print("Running mini_nn examples\n")
    run_xor()
    print()
    run_regression()
    print()
    run_iris_like()


if __name__ == "__main__":
    main()
