''' This module contains abstract base classes for the various neural network algorithms. '''


import numpy

from nlplib.core.control.neuralnetwork.collection import Array, Matrix
from nlplib.core.base import Base
from nlplib.general import math, composite

__all__ = ['Feedforward', 'Backpropagate']

class Feedforward (Base) :
    ''' A pure Python implementation of the feedforward neural network algorithm. '''

    def __init__ (self, structure, input_indexes, active=1.0, inactive=0.0, activation=math.tanh) :

        self.structure = structure

        self.input_indexes = input_indexes

        self.active   = active
        self.inactive = inactive

        self.activation = activation

    def __call__ (self) :
        self._reset_input_charges()
        return self._feedforward()

    def _feedforward (self) :
        for inputs, connection, outputs in self.structure :
            outputs._charges = Array(self.activation(charge) for charge in self._excitement(connection, inputs))

        return outputs

    def _excitement (self, connection, charges) :
        for weights in connection :
            yield sum(weight * charge for weight, charge in zip(weights, charges))

    def _reset_input_charges (self) :
        inputs = self.structure.inputs()

        values = [self.inactive] * len(inputs)
        for index in self.input_indexes :
            values[index] = self.active

        inputs._charges = Array(values)

class Backpropagate (Base) :
    ''' A pure Python implementation of the backpropagation neural network training algorithm. '''

    def __init__ (self, structure, input_indexes, output_indexes, rate=0.2, activation_derivative=math.dtanh) :

        self.structure = structure

        self.input_indexes  = input_indexes
        self.output_indexes = output_indexes

        self.rate = rate
        self.activation_derivative = activation_derivative

    def __call__ (self) :
        differences = self._output_errors()
        self._hidden_errors()
        self._update_connection_weights()
        return self._total_error(differences)

    def _total_error (self, differences) :
        return sum(0.5 * difference ** 2 for difference in differences)

    def _correct_output_charges (self) :
        correct = [0.0] * len(self.structure.outputs())

        for index in self.output_indexes :
            correct[index] = 1.0

        return correct

    def _output_errors (self) :

        correct = self._correct_output_charges()

        outputs = self.structure.outputs()

        differences = [correct_charge - actual_charge for correct_charge, actual_charge in zip(correct, outputs)]

        outputs._errors = Array(self.activation_derivative(charge) * difference
                                for charge, difference in zip(outputs, differences))

        return differences

    def _excitement_error (self, connection, errors) :
        for weights in connection.weights.transpose() :
            yield sum(weight * charge for weight, charge in zip(weights, errors))

    def _hidden_errors (self) :
        for inputs, connection, outputs in reversed(list(self.structure)[1:]) :

            errors = list(self._excitement_error(connection, outputs.errors))

            inputs._errors = Array(self.activation_derivative(charge) * error
                                   for charge, error in zip(inputs.charges, errors))

    def _update_connection_weights (self) :

        for inputs, connection, outputs in reversed(self.structure) :

            corrections = ((self.rate * charge * error for charge in inputs.charges)
                           for error in outputs.errors)

            connection._weights = Matrix(tuple(weight + correction_weight
                                               for weight, correction_weight in zip(weights, correction_weights))
                                         for weights, correction_weights in zip(connection, corrections))

def __test__ (ut, feedforward_cls=Feedforward, backpropagate_cls=Backpropagate) :
    import random

    from nlplib.core.model import NeuralNetwork

    random.seed(0)

    nn = NeuralNetwork('abc', 3, 'def')

    def train (inputs, outputs) :

        feedforward_cls(nn._structure, nn._structure.input_indexes_for_objects(inputs))()

        return backpropagate_cls(nn._structure, nn._structure.input_indexes_for_objects(inputs),
                                 nn._structure.output_indexes_for_objects(outputs))()

    training_patterns = [('ab', 'f'),
                         ('ac', 'f'),
                         ('b',  'd'),
                         ('c',  'e')]

    ut.assert_equal('%.8f' % sum(train(inputs, outputs) for _ in range(200) for inputs, outputs in training_patterns),
                    '19.47553614')

    correct_outputs = [[('0.97021362', 'f'), ('-0.84882957', 'e'), ('-0.50709088', 'd')],
                       [('0.92238128', 'd'), ('0.00134225', 'e'),  ('-0.00298671', 'f')],
                       [('0.93247123', 'e'), ('0.00284603', 'f'),  ('0.00172951', 'd')],
                       [('0.97872110', 'f'), ('0.07496323', 'd'),  ('-0.03463506', 'e')],
                       [('0.98019596', 'f'), ('0.01948382', 'e'),  ('-0.04068978', 'd')],
                       [('0.90101923', 'd'), ('0.63745278', 'e'),  ('-0.01855872', 'f')]]

    for inputs, correct_output in zip(['a', 'b', 'c', 'ab', 'ac', 'bc'], correct_outputs) :
        output_layer = feedforward_cls(nn._structure, nn._structure.input_indexes_for_objects(inputs))()

        actual_output = [('%.8f' % charge, object)
                         for object, charge in zip(output_layer.objects(), output_layer.charges)]

        ut.assert_equal(sorted(actual_output, reverse=True), correct_output)

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

