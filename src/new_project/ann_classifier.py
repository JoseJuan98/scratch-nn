# -*- utf-8 -*-
"""Artificial Neural Network Classifier.

This module implements an artificial neural network for classification
"""
from typing import List, Dict, Callable, Tuple, Union

import numpy

from matplotlib import pyplot
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm

from .ann import ANN

from .ann_functions import softmax, relu, relu_derivative, cross_entropy_loss, cross_entropy_loss_derivative


class ANNClassifier(ANN):
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
    activation_functions: List[Callable[[numpy.ndarray], numpy.ndarray]] = []
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
        activations: List[Callable[[numpy.ndarray], numpy.ndarray]],
        derivated_activations: List[Union[
            Callable[[numpy.ndarray], numpy.ndarray],
            Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray]
        ]],
        loss_function: Callable[[numpy.ndarray, numpy.ndarray], float],
        learning_rate: float = 0.001,
    ):
        """Instantiate an ANNClassifier object.

        It also initilizes the weights and biases.
        """
        super().__init__(
            layer_sizes=layer_sizes,
            activations=activations,
            loss_function=loss_function,
            learning_rate=learning_rate,
            derivated_activations=derivated_activations
        )

    def predict(self, X: numpy.ndarray) -> numpy.ndarray:
        pass

    def forward_propagation(self, X: numpy.ndarray) -> numpy.ndarray:

        # self.parameters["Z_1"] = numpy.dot(X, self.weights["W_1"])
        # self.parameters["A_1"] = self.activation_functions[1](self.parameters["Z_1"])
        self.parameters[f"A_0"] = X

        # 2 ... nlayers
        for idx in range(1, self.n_layers):
            self.parameters[f"Z_{idx}"] = numpy.dot(self.parameters[f"A_{idx - 1}"], self.weights[f"W_{idx}"])
            self.parameters[f"A_{idx}"] = self.activation_functions[idx](self.parameters[f"Z_{idx}"])

        y_pred = self.parameters[f"A_{self.n_layers - 1}"]
        return y_pred

    def backward_propagation(self, y_pred: numpy.ndarray, Y: numpy.ndarray) -> None:

        self.parameters[f"dZ_{self.n_layers - 1}"] = self.derivated_activations[self.n_layers](y_pred, Y)
        self.parameters[f"dW_{self.n_layers - 1}"] = numpy.dot(
            self.parameters[f"A_{self.n_layers - 2}"].T, self.parameters[f"dZ_{self.n_layers - 1}"]
        )

        # nlayers-1 ... 1
        for idx in range(self.n_layers - 2, 0, -1):
            self.parameters[f"dZ_{idx}"] = (
                numpy.dot(
                    self.parameters[f"dZ_{idx + 1}"], self.parameters[f"dW_{idx + 1}"].T
                ) * self.derivated_activations[idx](self.parameters[f"Z_{idx}"])
            )
            self.parameters[f"dW_{idx}"] = numpy.dot(
                self.parameters[f"A_{idx - 1}"].T, self.parameters[f"dZ_{idx}"]
            )

    def update_parameters(self) -> None:

        for idx in range(1, self.n_layers):
            self.weights[f"W_{idx}"] -= self.learning_rate * self.parameters[f"dW_{idx}"]

    @staticmethod
    def _split_dataset(
        train_dataset: VisionDataset, percentage_train_size: float = 0.8
    ) -> Tuple[DataLoader, DataLoader]:
        # Define the mini-batch size
        batch_size = 1000

        if percentage_train_size > 1. or percentage_train_size < 0.:
            raise ValueError(f"Constraint 0 > percentange_train_size < 1 . Got instead {percentage_train_size}")

        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, validation_dataset = random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, validation_loader

    def train(self, train_dataset: VisionDataset, percentage_train_size: float = 0.8, epochs: int = 10):

        train_loader, validation_loader = self._split_dataset(
            train_dataset=train_dataset, percentage_train_size=percentage_train_size
        )

        device = "cpu"
        for epoch in range(epochs):
            # Training by looping over a training set
            train_loss_epoch = 0
            train_loop = tqdm(train_loader)
            for inputs, labels in train_loop:
                train_loop.set_description(desc=f"Epoch [{f'Epoch {epoch + 1}/{epochs}': <11}]")

                # Prepare data
                # TODO
                X = inputs.to(device).view(inputs.shape[0], self.layer_sizes[0]).numpy()
                labels = labels.to(device).cpu().numpy()

                # Convert labels to one-hot encoding
                Y = numpy.eye(self.layer_sizes[-1])[labels]
                del inputs, labels

                # Forward propagation
                y_pred = self.forward_propagation(X)

                # Compute loss
                loss = self.loss_function(y_pred, Y)

                # when values are very small, in the derivated cross entropy there can be NaN values
                if loss is numpy.nan:
                    loss = 0.

                train_loss_epoch += loss
                self.train_losses.append(loss)

                # Backward propagation
                self.backward_propagation(y_pred=y_pred, Y=Y)

                # Update parameters
                self.update_parameters()

            train_loss_epoch /= len(train_loader)

            # Evaludate validation dataset
            val_accuracy, val_loss = self.evaluate(test_or_validation_loader=validation_loader, device=device)

            print(f"{f' ': <11} =>\t{f'Train Loss: {train_loss_epoch:.4f}': <20}\t"
                  f"{f'Validation loss: {val_loss:.4f}': <25}\tValidation Accuracy: {val_accuracy:.4f}")

    def evaluate(self, test_or_validation_loader: DataLoader, device: str = "cpu") -> Tuple[float, float]:
        """Method used to perform evaluation on the test or the validation datasets."""
        evaluation_loss = 0
        correct = 0
        total = 0

        for inputs, labels in test_or_validation_loader:
            # Prepare data
            # TODO
            X = inputs.to(device).view(inputs.shape[0], self.layer_sizes[0]).numpy()
            labels = labels.to(device).cpu().numpy()

            # Convert labels to one-hot encoding
            Y = numpy.eye(self.layer_sizes[-1])[labels]
            del inputs, labels

            # Forward propagation
            y_pred = self.forward_propagation(X)

            # Compute loss
            loss = self.loss_function(y_pred, Y)

            # when values are very small, in the derivated cross entropy there can be NaN values
            if loss is numpy.nan:
                loss = 0.
            # Calculate accuracy
            predicted_labels = numpy.argmax(y_pred, axis=1)
            correct += numpy.sum(predicted_labels.reshape(predicted_labels.shape[0], 1) == Y)
            total += Y.shape[0]

        evaluation_loss /= len(test_or_validation_loader)

        self.val_losses.append(evaluation_loss)
        evaluation_accuracy = correct / total

        return evaluation_accuracy, evaluation_loss

    def plot_train_validation_loss(self) -> None:
        pyplot.plot(range(1, len(self.train_losses) + 1), self.train_losses, label="Training Loss")
        pyplot.plot(range(1, len(self.val_losses) + 1), self.val_losses, label="Validation Loss")
        pyplot.xlabel("Epochs")
        pyplot.ylabel("Loss")
        pyplot.title("Train vs Validation Loss")
        pyplot.legend()
        pyplot.show();  # noqa

