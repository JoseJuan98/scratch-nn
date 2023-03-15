#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __________________________________________
#
# Version: 2.0
# Author: Jose Pena
# Date: 1/2023
#
# __________________________________________
#
"""Base Neural Network """

from abc import ABC, abstractmethod
from numpy import ndarray


class NeuralNet(ABC):
    """
    Abstract class to implement different NeuralNetwork Architectures as a python class
    """

    @abstractmethod
    def backward_propagation(self, AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        """

    @abstractmethod
    def forward_propagation(self, X, parameters):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
        """

    @abstractmethod
    def compute_cost(self, AL, Y):
        """
        Cost function

        Returns:
            numpy.array: cost value as numpy array of dimension 0, not scalar
        """

    @abstractmethod
    def train(self,
              X: ndarray,
              Y: ndarray,
              learning_rate: float = 0.0075,
              num_iterations: int = 3000):
        """
        Performs the computation of the weights of the neurons

        Args:
            X (numpy.ndarray): dependent variables' values
            Y (numpy.ndarray): independent variable's values
            learning_rate (float): learning rate use for gradient descent (C)
            num_iterations (int): number of iterations to compute back and fordward propagation

        Returns:
            self
        """

    @abstractmethod
    def predict(self) -> ndarray:
        """
        Performs the prediction with the trained neurons

        Returns:
            numpy.array: prediction values
        """
