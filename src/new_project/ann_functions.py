# -*- utf-8 -*-
"""Artificial Neural Network functions.

This module contains functions which are NumPy implementation of equations or formulas used to work with Artificial
Neural Networks. For example, the CrossEntropyLoss function.
"""
from typing import Union

import numpy


def relu(x: numpy.ndarray) -> numpy.ndarray:
    """ReLU activation function."""
    return numpy.maximum(0, x)


def relu_derivative(x: numpy.ndarray) -> numpy.ndarray:
    """Derivative of the ReLU activation function."""
    return (x > 0).astype(x.dtype)


def softmax(x: numpy.ndarray) -> numpy.ndarray:
    """Softmax activation function."""
    # to avoid the vanishing gradient problem
    x = numpy.clip(x, -50000, 50000)  # to handle RuntimeWarning: invalid value encountered in subtract
    exp_x = numpy.exp(x - numpy.max(x, axis=1, keepdims=True))
    return exp_x / numpy.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(y_pred: numpy.ndarray, y_true: numpy.ndarray) -> Union[numpy.ndarray, float]:
    """Cross Entropy Loss function."""
    epsilon = 1e-8  # Small constant to avoid log(0) , to handle (RuntimeWarning: divide by zero encountered in
                    # log and RuntimeWarning: invalid value encountered in multiply)
    return -numpy.sum(y_true * numpy.log(y_pred + epsilon)) / y_true.shape[0]


def cross_entropy_loss_derivative(y_pred: numpy.ndarray, y_true: numpy.ndarray) -> numpy.ndarray:
    """Derivative of the Cross Entropy Loss function with respect softmax.

    Equation:
        $$\frac{\partial L}{\partial \hat{y}} = \hat{y} - y$$
    """
    return y_pred - y_true

# Clip gradients for  RuntimeWarning: invalid value encountered in multiply
def clip_gradients( gradients, clip_value):
    for i in range(len(gradients)):
        numpy.clip(gradients[i], -clip_value, clip_value, out=gradients[i])
    return gradients
