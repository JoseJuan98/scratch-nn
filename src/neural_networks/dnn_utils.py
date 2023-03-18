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
""" Utilities for working with Deep Neural Network """
from typing import Any, Union, Tuple

import numpy


def sigmoid(Z: Union[numpy.ndarray[Any, numpy.dtype[Any]], int, float]
            ) -> Tuple[Union[numpy.ndarray[Any, numpy.dtype[Any]], int, float],
                       Union[numpy.ndarray[Any, numpy.dtype[Any]], int, float]]:
    """
    Sigmoid activation function.

    Args:
        Z (numpy.array, int, float): input of the sigmoid function. Scalar or vector.

    Returns:
        (numpy.array, int, float): result of the function sigmoid(z), same shape as Z
        (numpy.array, int, float): returns Z to later be used on backpropagation
    """
    return 1 / (1 + numpy.exp(-Z)), Z


def relu(Z):
    """
    RELU function.

    Args:
        Z: output of the linear layer, of any shape

    Returns:
        A: Post-activation parameter, same shape as Z
        cache: dictionary containing "A" ; stored for computing the backward propagation efficiently
    """

    A = numpy.maximum(0, Z)

    assert (A.shape == Z.shape), "Post-activation parameter doesn't have same shape as Z"

    cache = Z
    return A, cache


def relu_backward(dA, cache):
    """
    Backward propagation for a single RELU unit.

    Args:
        dA: post-activation gradient, of any shape
        cache: 'Z', stored for computing backward propagation efficiently

    Returns:
        dZ: gradient of the cost with respect to Z
    """

    Z = cache
    dZ = numpy.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape), "dZ, Gradient descent of Z, doesn't have same shape as Z"

    return dZ


def sigmoid_backward(dA, cache):
    """
    Backward propagation for a single SIGMOID unit.

    Args:
        dA: post-activation gradient, of any shape
        cache: 'Z', stored for computing backward propagation efficiently

    Returns:
        dZ: gradient of the cost with respect to Z
    """

    Z = cache

    s = 1 / (1 + numpy.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ
