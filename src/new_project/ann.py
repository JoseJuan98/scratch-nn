# -*- utf-8 -*-
"""Artificial Neural Network.

This module implements the abstract class of an artificial neural network.
"""
from typing import List, Dict, Callable, Union, Tuple
from abc import ABC, abstractmethod

import numpy

from torch.utils.data import DataLoader


class ANN(ABC):
    """Artificial Neural Network.

    Implementation of the Multiple-Layers Perceptron (ANN), which can be specified the amount and depth of layers
    and its activation functions a long side other parameters like learning rate.

    Attributes:


    """
    learning_reate: float = None
    layer_sizes: List[int] = []
    weights: Dict[str, numpy.ndarray] = {}
    biases: Dict[str, numpy.ndarray] = {}
    parameters: Dict[str, numpy.ndarray] = {}
    activation_functions: List[Union[Callable[[numpy.ndarray], numpy.ndarray], None]] = []
    derivated_activations: List[Union[
        Callable[[numpy.ndarray], numpy.ndarray],
        Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray],
        None
    ]] = []
    loss_function: Callable[[numpy.ndarray, numpy.ndarray], float] = None
    train_losses: List[float] = []
    val_losses: List[float] = []
    n_layers: int

    def __init__(
        self,
        layer_sizes: List[int],
        activations: List[Union[Callable[[numpy.ndarray], numpy.ndarray], None]],
        derivated_activations: List[Union[
            Callable[[numpy.ndarray], numpy.ndarray],
            Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray],
            None
        ]],
        loss_function: Callable[[numpy.ndarray, numpy.ndarray], float],
        learning_rate: float = 0.001,
    ):
        """Instantiate an ANN object."""
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.train_losses = []
        self.val_losses = []
        self._initiliaze_parameters()
        self.parameters = {}
        self.n_layers = len(self.layer_sizes)

        if len(activations) != len(self.layer_sizes):
            raise ValueError(f"The number of activation functions must be the same as the number of layers.\n"
                             f"Got instead {(len(activations), len(self.layer_sizes))} - (nactivations, nlayers)")

        if len(activations) != len(self.layer_sizes):
            raise ValueError(f"The number of activation functions must be the same as the number of layers.\n"
                             f"Got instead {(len(activations), len(self.layer_sizes))} - (nactivations, nlayers)")

        # all indexes starts by 1, so the first element must be none
        self.activation_functions.append(None)
        self.derivated_activations.append(None)
        for activation in activations:
            self.activation_functions.append(activation)

        for dactivation in derivated_activations:
            self.derivated_activations.append(dactivation)

    def _initiliaze_parameters(self):
        """Initialize the weights and biases based in the sizes of the layers."""

        # for idx in range(1, self.n_layers):
        for idx, size in enumerate(self.layer_sizes[1:], start=1):
            self.weights[f"W_{idx}"] = numpy.random.randn(self.layer_sizes[idx - 1], size) * 0.01
            self.biases[f"b_{idx}"] = numpy.zeros((size, 1))

            if self.weights[f"W_{idx}"].shape != (self.layer_sizes[idx -1], size):
                raise ValueError("The weights must have the same shape as the input layer's ")

            if self.biases[f"b_{idx}"].shape != (size, 1):
                raise ValueError("The weights must have the same shape as the input layer's ")

    @abstractmethod
    def predict(self, X: numpy.ndarray) -> numpy.ndarray:
        pass

    @abstractmethod
    def forward_propagation(self, X: numpy.ndarray) -> numpy.ndarray:
        pass

    @abstractmethod
    def backward_propagation(self, y_pred: numpy.ndarray, Y: numpy.ndarray) -> None:
        pass

    @abstractmethod
    def update_parameters(self):
        pass

    @abstractmethod
    def train(self, train_loader: DataLoader, validation_loader: DataLoader, epochs: int = 10):
        """Train the model on the given train and validation dataset"""
        pass

    @abstractmethod
    def evaluate(self, test_or_validation_loader: DataLoader, device: str = "cpu") -> Tuple[float, float]:
        pass

    @abstractmethod
    def plot_train_validation_loss(self) -> None:
        pass
