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

import numpy


def sigmoid(Z):
    """
    Sigmoid activation function.

    Args:
        Z: numpy array of any shape

    Returns:
        A: result of the function sigmoid(z), same shape as Z
        cache: returns Z to later be used on backpropagation
    """
    A = 1 / (1 + numpy.exp(-Z))
    cache = Z

    return A, cache


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

    assert (A.shape == Z.shape)

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

    assert (dZ.shape == Z.shape)

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
