''' This module contains abstract base classes for the various neural network algorithms. '''


from nlplib.core.base import Base
from nlplib.general import math

__all__ = ['Feedforward', 'Backpropagate']

class Feedforward (Base) :
    def __init__ (self, structure, input_indexes, active=1.0, inactive=0.0, activation=math.tanh) :

        self.structure = structure

        self.input_indexes = input_indexes

        self.active   = active
        self.inactive = inactive

        self.activation = activation

    def __call__ (self) :

        self._reset_input_charges()

        for input_layer, connection, output_layer in self.structure :
            output_layer._charges = tuple(self.activation(charge)
                                          for charge in self._excitement(connection, input_layer))
        return output_layer

    def _excitement (self, connection, charges) :
        for weights in connection :
            yield sum(weight * charge for weight, charge in zip(weights, charges))

    def _reset_input_charges (self) :
        inputs = self.structure.inputs()
        inputs.fill(self.inactive)
        inputs.set((index, self.active) for index in self.input_indexes)

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

        total_error = sum(0.5 * difference ** 2 for difference in differences)
        return total_error

    def _correct_output_charges (self) :
        correct = [0.0] * len(self.structure.outputs())

        for index in self.output_indexes :
            correct[index] = 1.0

        return correct

    def _output_errors (self) :

        correct = self._correct_output_charges()

        outputs = self.structure.outputs()

        differences = [correct_charge - actual_charge for correct_charge, actual_charge in zip(correct, outputs)]

        outputs._errors = tuple(self.activation_derivative(charge) * difference
                                for charge, difference in zip(outputs, differences))

        return differences

    def _excitement_error (self, connection, errors) :
        transposed = zip(*connection.weights)
        for weights in transposed :
            yield sum(weight * charge for weight, charge in zip(weights, errors))

    def _hidden_errors (self) :
        for input_layer, connection, output_layer in reversed(list(self.structure)[1:]) :

            errors = self._excitement_error(connection, output_layer.errors)

            input_layer._errors = tuple(self.activation_derivative(charge) * error
                                        for charge, error in zip(input_layer.charges, errors))

    def _update_connection_weights (self) :

        for input_layer, connection, output_layer in reversed(self.structure) :

            corrections = ((self.rate * charge * error for charge in input_layer.charges)
                            for error in output_layer.errors)

            connection._weights = tuple(tuple(weight + correction_weight
                                              for weight, correction_weight in zip(weights, correction_weights))
                                        for weights, correction_weights in zip(connection, corrections))

def __test__ (ut) :
    from nlplib.core.model import NeuralNetwork

    nn = NeuralNetwork('abc', 3, 'def')

    def train (inputs, outputs) :

        Feedforward(nn._structure, nn._structure.input_indexes_for_objects(inputs))()

        return Backpropagate(nn._structure, nn._structure.input_indexes_for_objects(inputs),
                             nn._structure.output_indexes_for_objects(outputs))()

    training_patterns = [('ab', 'f'),
                         ('ac', 'f'),
                         ('b',  'd'),
                         ('c',  'e')]

    from nlplib.exterior.util import plot

    for _ in range(20) :
        for inputs, outputs in training_patterns :
            train(inputs, outputs)

    for inputs in ['a', 'b', 'c', 'ab', 'ac', 'bc'] :
        output_layer = Feedforward(nn._structure, nn._structure.input_indexes_for_objects(inputs))()

        print(sorted((item for item in zip(output_layer.charges, output_layer.objects())), reverse=True))

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

