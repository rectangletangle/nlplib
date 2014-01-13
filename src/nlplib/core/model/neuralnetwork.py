

import itertools

from nlplib.core.control.neuralnetwork.structure import MakeStructure, random_weights
from nlplib.core.control.score import Score
from nlplib.core.model.naturallanguage import Seq
from nlplib.core.model.base import Model
from nlplib.core.base import Base
from nlplib.general.iterate import truncated, paired
from nlplib.general import math

try :
    # An attempt is made to import accelerated versions of the neural network algorithms that utilize NumPy.
    from nlplib.core.control.neuralnetwork.numpy_ import Array, Matrix, Feedforward, Backpropagate
except ImportError :
    # Fall back to the slower pure Python versions, if NumPy isn't installed.
    from nlplib.core.control.neuralnetwork import Array, Matrix, Feedforward, Backpropagate

__all__ = ['NeuralNetwork', 'Structure', 'Element', 'Layer', 'Connection', 'NeuralNetworkIO']

class NeuralNetwork (Model) :
    def __init__ (self, *config, name=None, weights=random_weights) :
        self.name = name

        self._structure = Structure(config, weights=weights)

    def __repr__ (self, *args, **kw) :
        if self.name is not None :
            return super().__repr__(self.name, *args, **kw)
        else :
            return super().__repr__(*args, **kw)

    def __contains__ (self, object) :
        return object in tuple(self)

    def __iter__ (self) :
        return itertools.chain(self.inputs(), self.outputs())

    def inputs (self) :
        yield from self._structure.inputs().objects()

    def outputs (self) :
        yield from self._structure.outputs().objects()

    def scores (self) :
        for object, charge in zip(self._structure.outputs().objects(), self._structure.outputs()) :
            yield Score(object, score=charge)

    def predict (self, input_objects, *args, **kw) :
        input_indexes = self._structure.input_indexes_for_objects(input_objects)

        self._structure.feedforward(input_indexes, *args, **kw)

        yield from self.scores()

    def train (self, input_objects, output_objects, *args, **kw) :

        input_indexes  = self._structure.input_indexes_for_objects(input_objects)
        output_indexes = self._structure.output_indexes_for_objects(output_objects)

        return self._structure.backpropogate(input_indexes, output_indexes, *args, **kw)

    def forget (self) : # todo :
        ''' This resets the network to an untrained state. '''

        raise NotImplementedError

    def _associated (self, session) :
        yield self._structure

class Structure (Model) :
    def __init__ (self, config, weights) :
        self.elements = []

        self.layers      = []
        self.connections = []

        if len(config) :
            MakeStructure(self, config, weights)()

    def __iter__ (self) :
        for (input_layer, output_layer), connection in zip(paired(self.layers), self.connections) :
            yield (input_layer, connection, output_layer)

    def __reversed__ (self) :
        return reversed(list(self))

    def input_indexes_for_objects (self, objects) :
        # todo : this could be done more efficiently
        input_objects = list(self.inputs().objects())
        for object in objects :
            yield input_objects.index(object)

    def output_indexes_for_objects (self, objects) :
        # todo : this could be done more efficiently
        input_objects = list(self.outputs().objects())
        for object in objects :
            yield input_objects.index(object)

    def inputs (self) :
        return self.layers[0]

    def outputs (self) :
        return self.layers[-1]

    def hidden (self, reverse=False) :
        layers = self if not reverse else reversed(self.layers)
        return truncated(itertools.islice(layers, 1, None), 1)

    def feedforward (self, input_indexes, *args, **kw) :
        return Feedforward(self, input_indexes, *args, **kw)()

    def backpropogate (self, input_indexes, output_indexes, rate=0.2, activation_derivative=math.dtanh,
                       **kw) :
        ''' This method allows for supervised training, using the backpropagation algorithm. '''

        self.feedforward(input_indexes, **kw)

        return Backpropagate(self, input_indexes, output_indexes, rate=rate,
                             activation_derivative=activation_derivative)()

    def clear (self) :
        ''' This deletes all of the network's nodes, leaving an empty network. '''

        raise NotImplementedError # todo :

    def _associated (self, session) :
        return self.elements

class Element (Model) :
    ''' This is a base class structural neural network elements, e.g., links, nodes, or IO nodes. '''

    def __init__ (self, structure) :
        self.structure = structure
        self.structure.elements.append(self)

class Layer (Element) :
    def __init__ (self, structure) :
        super().__init__(structure)
        self.structure.layers.append(self)

        self._charges = Array()
        self._errors  = Array()

        self.io = []

    def __repr__ (self, *args, **kw) :
        return super().__repr__(len(self._charges), *args, **kw)

    def __iter__ (self) :
        return iter(self._charges)

    def __len__ (self) :
        return len(self._charges)

    @property
    def charges (self) :
        return self._charges

    @property
    def errors (self) :
        return self._errors

    def objects (self) :
        for io in self.io :
            yield io.object

    def _associated (self, session) :
        return self.io

class Connection (Element) :
    def __init__ (self, structure) :
        super().__init__(structure)
        self.structure.connections.append(self)

        self._weights = Matrix()

    def __iter__ (self) :
        return iter(self._weights)

    def __repr__ (self, *args, **kw) :
        width  = self._weights.width()
        height = self._weights.height()

        # A mock "dimensions" object for the textual representation.
        dimensions = type('Dimensions', (), {'__repr__' : lambda self : '{}x{}'.format(width, height)})()

        return super().__repr__(dimensions, *args, **kw)

    @property
    def weights (self) :
        return self._weights

class NeuralNetworkIO (Element) :
    def __init__ (self, structure, object) :
        super().__init__(structure)

        self.object = object

    def __repr__ (self, *arg, **kw) :
        return super().__repr__(self.object, *arg, **kw)

    def _make_object (self) :
        self._object = self._model if self._model is not None else self._json

    @property
    def object (self) :
        return self._object

    @object.setter
    def object (self, value) :
        self._model = None
        self._json  = None

        if isinstance(value, Seq) : # todo : make handle all models not just Seq
            self._model = value
        else :
            self._json = value

        self._object = value

