''' This module contains basic neural network algorithms implemented using the NumPy library. This allows for
    drastically improved performance over the pure Python implementation. '''


import numpy

from nlplib.core.control import neuralnetwork as python
from nlplib.general import composite

__all__ = ['Array', 'Matrix', 'Feedforward', 'Backpropagate']

class Array (python.Array) :

    __slots__ = ('_values',)

    def __init__ (self, values=()) :
        self._values = numpy.array(list(values))

class Matrix (python.Matrix) :

    __slots__ = ('_values',)

    def __init__ (self, values=()) :
        self._values = numpy.matrix(list(values))

    def __iter__ (self) :
        for row in self._values :
            yield tuple(row.flat)

    def transpose (self) :
        matrix = self.__class__.__new__(self.__class__)
        matrix._values = self._values.transpose()
        return matrix

    def width (self) :
        return self._values.shape[1]

    def height (self) :
        return self._values.shape[0]

class Feedforward (python.Feedforward) :
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

class Backpropagate (python.Backpropagate) :
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

        # Due to a quirk in SQLAlchemy, mutations may not get written to the database. Reassigning the name, makes sure
        # that the mutations are recorded.
        outputs._errors = outputs._errors

        return differences

    def _excitement_error (self, connection, errors) :
        return numpy.asarray(numpy.dot(connection._weights._values.transpose(), errors._values)).flatten()

    def _hidden_errors (self) :

        activation_derivative = self._vectorized_activation_derivative

        for inputs, connection, outputs in reversed(list(self.structure)[1:]) :

            errors = self._excitement_error(connection, outputs.errors)

            inputs._errors._values[:] = activation_derivative(inputs._charges._values) * errors

            # This step is necessary, see the comment in <Backpropagate._output_errors>.
            inputs._errors = inputs._errors

    def _update_connection_weights (self) :
        for inputs, connection, outputs in reversed(self.structure) :
            weights = self.rate * (inputs._charges._values * outputs._errors._values[:,numpy.newaxis])

            connection._weights._values += weights

            # This step is necessary, see the comment in <Backpropagate._output_errors>.
            connection._weights = connection._weights

def __test__ (ut) :
    from nlplib.core.control.neuralnetwork import _test_collections, _test_feedforward_and_backpropagate

    # The same tests work for the NumPy neural network algorithms.
    _test_collections(ut, array_cls=Array, matrix_cls=Matrix)
    _test_feedforward_and_backpropagate(ut, feedforward_cls=Feedforward, backpropagate_cls=Backpropagate)

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

