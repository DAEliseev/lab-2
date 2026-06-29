# Лабораторная работа по курсу "Искусственный интеллект"
# Создание своего нейросетевого фреймворка

### Студенты: 

| ФИО       | Роль в проекте                     | Оценка       |
|-----------|------------------------------------|--------------|
| Елисеев   | Разработчик                        |              |

## Описание

`mini_nn` - учебный нейросетевой фреймворк для полносвязных сетей, написанный на Python.

## Структура проекта

```text
mini_nn/
  datasets.py    # Dataset API
  layers.py      # Dense и функции активации
  losses.py      # функции потерь
  model.py       # Sequential и цикл обучения
  optimizers.py  # SGD, MomentumSGD, GradientClippingSGD
  tensor.py      # операции над списками и параметры
examples/
  xor.py
  regression.py
  iris_like.py
main.py          # запуск всех примеров
```

## Быстрый старт

```bash
python main.py
```

Запуск отдельных примеров:

```bash
python -m examples.xor
python -m examples.regression
python -m examples.iris_like
```

## Пример использования

```python
from mini_nn import BinaryCrossEntropy, Dataset, Dense, MomentumSGD, Sequential, Sigmoid, Tanh

dataset = Dataset([
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0]),
])

model = Sequential([
    Dense(2, 4, seed=1),
    Tanh(),
    Dense(4, 1, seed=2),
    Sigmoid(),
])

model.fit(
    dataset,
    loss=BinaryCrossEntropy(),
    optimizer=MomentumSGD(lr=0.5, momentum=0.8),
    epochs=1200,
    batch_size=4,
)

print(model.predict([1.0, 0.0]))
```

## Dataset API

`Dataset` хранит пары `(inputs, target)`, где оба значения являются списками чисел. Основные методы:

- `Dataset.from_lists(inputs, targets)` - создать датасет из двух списков.
- `dataset.map(func)` - применить функцию к каждому примеру.
- `dataset.shuffle(seed=None)` - вернуть перемешанную копию.
- `dataset.minibatches(batch_size)` - получить mini-batch-и для обучения.
- `dataset.normalize()` - стандартизировать входные признаки.
- `train_test_split(dataset, test_ratio=0.2)` - разделить датасет на train/test.

## Как устроено обучение

Каждый слой реализует два метода:

- `forward(inputs)` считает выход слоя.
- `backward(grad_output)` возвращает градиент по входу и накапливает градиенты параметров.

`Sequential.fit` проходит по mini-batch-ам, считает loss, запускает backpropagation и передаёт параметры выбранному оптимизатору.

## Реализованные примеры

- `examples/xor.py` - бинарная классификация XOR.
- `examples/regression.py` - аппроксимация нелинейной функции с MSE.
- `examples/iris_like.py` - классификация цветов по четырём признакам и трём классам в стиле Iris.
