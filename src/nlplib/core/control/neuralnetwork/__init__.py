''' This module contains abstract base classes for the various neural network algorithms. '''


from nlplib.core.base import Base
from nlplib.general import math, composite

__all__ = ['Array', 'Matrix', 'Feedforward', 'Backpropagate']

class Array (Base) :

    __slots__ = ('_values',)

    def __init__ (self, values=()) :
        self._values = tuple(values)

    def __repr__ (self, *args, **kw) :
        return super().__repr__(list(self._values), *args, **kw)

    def __iter__ (self) :
        return iter(self._values)

    def __eq__ (self, other) :
        return list(self._values) == other

    def __len__ (self) :
        return len(self._values)

class Matrix (Base) :

    __slots__ = ('_values',)

    def __init__ (self, values=()) :
        self._values = list(values)

    def __repr__ (self, *args, **kw) :
        return super().__repr__(list(self), *args, **kw)

    def __eq__ (self, other) :
        return list(self) == other

    def __len__ (self) :
        return len(self._values)

    def __iter__ (self) :
        for row in self._values :
            yield tuple(row)

    def transpose (self) :
        return self.__class__(zip(*self._values))

    def width (self) :
        try :
            return len(self._values[0])
        except IndexError :
            return 0

    def height (self) :
        return len(self._values)

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

    def __init__ (self, structure, input_indexes, output_indexes, rate=0.2, active=1.0, inactive=0.0,
                  activation_derivative=math.dtanh) :

        self.structure = structure

        self.input_indexes  = input_indexes
        self.output_indexes = output_indexes

        self.rate = rate

        self.active   = active
        self.inactive = inactive

        self.activation_derivative = activation_derivative

    def __call__ (self) :
        differences = self._output_errors()
        self._hidden_errors()
        self._update_connection_weights()
        return self._total_error(differences)

    def _total_error (self, differences) :
        return sum(0.5 * difference ** 2 for difference in differences)

    def _correct_output_charges (self) :
        correct = [self.inactive] * len(self.structure.outputs())

        for index in self.output_indexes :
            correct[index] = self.active

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

def _test_collections (ut, array_cls=Array, matrix_cls=Matrix) :
    import pickle

    # Tests the array.
    array = array_cls([1, 2, 3])
    ut.assert_equal(len(array), 3)
    ut.assert_equal(list(array), [1, 2, 3])
    ut.assert_equal(array, [1, 2, 3])

    array = array_cls([])
    ut.assert_equal(list(array), [])
    ut.assert_equal(array, [])

    array = array_cls([2, 9, 4])
    ut.assert_equal(pickle.loads(pickle.dumps(array)), [2, 9, 4])

    # Tests the matrix.
    matrix = matrix_cls([(1, 2, 3), (4, 5, 6)])
    ut.assert_equal(matrix.width(), 3)
    ut.assert_equal(matrix.height(), 2)
    ut.assert_equal(len(matrix), 2)
    ut.assert_equal(list(matrix), [(1, 2, 3), (4, 5, 6)])

    matrix = matrix.transpose()
    ut.assert_equal(matrix.width(), 2)
    ut.assert_equal(matrix.height(), 3)
    ut.assert_equal(len(matrix), 3)
    ut.assert_equal(list(matrix), [(1, 4), (2, 5), (3, 6)])

    matrix = matrix.transpose()
    ut.assert_equal(matrix.width(), 3)
    ut.assert_equal(matrix.height(), 2)
    ut.assert_equal(list(matrix), [(1, 2, 3), (4, 5, 6)])

    ut.assert_equal(pickle.loads(pickle.dumps(matrix)), [(1, 2, 3), (4, 5, 6)])

def _test_feedforward_and_backpropagate (ut, feedforward_cls=Feedforward, backpropagate_cls=Backpropagate) :
    import random

    from nlplib.core.model import Database, NeuralNetwork

    random.seed(0)

    nn = NeuralNetwork('abc', 3, 'def', name='foo')

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

    def test_outputs (nn) :
        for inputs, correct_output in zip(['a', 'b', 'c', 'ab', 'ac', 'bc'], correct_outputs) :
            output_layer = feedforward_cls(nn._structure, nn._structure.input_indexes_for_objects(inputs))()

            actual_output = [('%.8f' % charge, object)
                             for object, charge in zip(output_layer.objects(), output_layer.charges)]

            ut.assert_equal(sorted(actual_output, reverse=True), correct_output)

    test_outputs(nn)

    # Test that all of the weights are being persisted properly
    db = Database()
    with db as session :
        session.add(nn)

    with db as session :
        nn_from_db = session.access.nn('foo')
        test_outputs(nn_from_db)

def __test__ (ut) :
    _test_collections(ut)
    _test_feedforward_and_backpropagate(ut)

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

