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
""" Unit test for dnn_utils.py"""

from unittest import TestCase

import numpy

from neural_networks.dnn_utils import sigmoid


class TestDNNUtils(TestCase):
    """
    # TODO
    """

    def test_sigmoid(self):
        self.assertEqual(sigmoid(Z=3), (0.9525741268224334, 3))

    def test_sigmoid_raise_type_error(self):
        with self.assertRaises(TypeError):
            sigmoid([1, 2, 3])  # type: ignore [arg-type]

    def test_sigmoid_numpy_array(self):
        x = numpy.array([1, 2, 3])
        A, cache = sigmoid(x)
        A = numpy.round(A, 8)
        self.assertEqual(A.shape, (3,))
        self.assertTrue(numpy.array_equal(A, numpy.array([0.73105858,
                                                          0.88079708,
                                                          0.95257413]
                                                         )
                                          )
                        )
