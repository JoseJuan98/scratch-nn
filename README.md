# Neural Network From Scratch

This repository contains a Python library for building simple Deep Neural Networks from scratch, using only vectorized operations with [NumPy](https://numpy.org/). 

The goal of this project is to develop further understanding in the inner workings of neural networks and provide a foundation for building a Production-ready Python library.

## Features

The current version of the library includes the following features:

-   A customizable neural network architecture with support for multiple layers and activation functions
-   Stochastic gradient descent optimization algorithm with support for different loss functions
[comment]<> : (
-   A suite of evaluation metrics, such as accuracy and mean squared error, for measuring model performance
-   A suite of utilities for data preprocessing, including normalization and one-hot encoding
)
## Getting started

To use this library, you need to install it on your system. You can install it by running 

```shell
pip install --no-cache-dir .
```

Once you have it installed, you can import the library into your Python code using:

```python
from neural_network import DeepNN
```

From there, you can create an instance of the Deep Neural Network architecture as a python class and customize it to fit your needs. For example, to create a neural network with one hidden layer and a sigmoid activation function, you can use the following code:

```python
nn = DeepNN(layers_dims=[input_size, hidden_size, output_size], activations=["sigmoid", "sigmoid"])
```

## Things to try

This library is a work in progress, and there are several areas where you can contribute and improve its functionality. Here are some ideas:

- [ ]   **Optimizing performance:** While the library is currently optimized for efficiency using NumPy, you can try using a GPU-accelerated computing library like CuPy to further improve its performance.
- [ ]   **Developing new architectures of NNs:** The current library supports only a simple feedforward neural network architecture, but you can explore other architectures, such as convolutional neural networks or recurrent neural networks, and implement them using the existing framework.
- [ ]   **Adding more evaluation metrics:** The library currently supports a limited set of evaluation metrics. You can add more metrics, such as precision and recall, to provide a more comprehensive view of model performance.

## Contributing

If you are interested in contributing to this project, please check out the [contribution guidelines](https://chat.openai.com/CONTRIBUTING.md) for more information on how to get started.

## License

This project is licensed under the [MIT License](https://chat.openai.com/LICENSE).