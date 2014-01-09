''' This module contains basic neural network algorithms implemented using the NumPy library. This allows for
    drastically improved performance over the pure Python implementation. '''


import numpy

from nlplib.core.control.neuralnetwork import Feedforward as NLPLibFeedforward, Backpropagate as NLPLibBackpropagate
from nlplib.general import composite

__all__ = ['Feedforward', 'Backpropagate']

class Feedforward (NLPLibFeedforward) :
    ''' A fast implementation of the feedforward neural network algorithm, using the NumPy library. '''

    def _feedforward (self) :
        activation = numpy.vectorize(self.activation)
        for inputs, connection, outputs in self.structure :
            outputs._charges._values[:] = activation(numpy.asarray(self._excitement(connection,
                                                                                    inputs.charges)).flatten())
            outputs._charges = outputs._charges

        return outputs

    def _excitement (self, connection, charges) :
        return numpy.dot(connection._weights._values, charges._values)

    def _reset_input_charges (self) :
        inputs = self.structure.inputs()

        inputs._charges._values.fill(self.inactive)
        inputs._charges._values[list(self.input_indexes)] = self.active
        inputs._charges = inputs._charges

class Backpropagate (NLPLibBackpropagate) :
    ''' A fast implementation of the backpropagation neural network training algorithm, using the NumPy library. '''

    def _total_error (self, differences) :
        return (0.5 * differences ** 2).sum()

    @composite(lambda self : (self.activation_derivative,))
    def _vectorized_activation_derivative (self) :
        return numpy.vectorize(self.activation_derivative)

    def _output_errors (self) :
        activation_derivative = self._vectorized_activation_derivative

        outputs = self.structure.outputs()

        correct = numpy.full(len(outputs), self.inactive)
        correct[list(self.output_indexes)] = self.active

        differences = correct - outputs._charges._values

        outputs._errors._values[:] = activation_derivative(outputs._charges._values) * differences
        outputs._errors = outputs._errors

        return differences

    def _excitement_error (self, connection, errors) :
        return numpy.asarray(numpy.dot(connection._weights._values.transpose(), errors._values)).flatten()

    def _hidden_errors (self) :

        activation_derivative = self._vectorized_activation_derivative

        for inputs, connection, outputs in reversed(list(self.structure)[1:]) :

            errors = self._excitement_error(connection, outputs.errors)

            inputs._errors._values[:] = activation_derivative(inputs._charges._values) * errors
            inputs._errors = inputs._errors

    def _update_connection_weights (self) :
        for inputs, connection, outputs in reversed(self.structure) :
            weights = self.rate * (inputs._charges._values * outputs._errors._values[:,numpy.newaxis])

            connection._weights._values += weights
            connection._weights = connection._weights

def __test__ (ut) :
    from nlplib.core.control.neuralnetwork import __test__

    return __test__(ut, feedforward_cls=Feedforward, backpropagate_cls=Backpropagate)

def __profile__ () :
    import random

    from nlplib.core.model import NeuralNetwork
    from nlplib.general import timing

    size  = 1000
    loops = 1

    random.seed(0)
    nn = NeuralNetwork(size, size, size)

    @timing
    def nlplib () :
        for _ in range(loops) :
            NLPLibBackpropagate(nn._structure, [0, 1, 2], [3, 4])()

    random.seed(0)
    nn = NeuralNetwork(size, size, size)

    @timing
    def numpy () :
        for _ in range(loops) :
            Backpropagate(nn._structure, [0, 1, 2], [3, 4])()

    nlplib()
    numpy()

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())
    __profile__()

