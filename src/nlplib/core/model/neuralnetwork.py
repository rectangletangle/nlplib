

import itertools
import random

from nlplib.core.control.neuralnetwork.alg import Feedforward, Backpropagate
from nlplib.core.control.neuralnetwork.structure import Static, StaticIO, _LayerConfiguration
from nlplib.core.model.naturallanguage import Seq
from nlplib.core.model.base import Model
from nlplib.core.control.score import Score
from nlplib.core.base import Base
from nlplib.core import exc
from nlplib.general.represent import pretty_float
from nlplib.general.iterate import truncated, paired, united
from nlplib.general import math

__all__ = ['NeuralNetwork', 'Layer', 'Connection', 'NeuralNetworkIO']

class NeuralNetwork (Model) :
    def __init__ (self, *config, name=None, **kw) :
        self.name = name

        self._structure = Structure(config, **kw)

    def __repr__ (self, *args, **kw) :
        if self.name is not None :
            return super().__repr__(self.name, *args, **kw)
        else :
            return super().__repr__(*args, **kw)

    def __contains__ (self) :
        raise NotImplementedError # todo : return bool() in object in inputs or outputs

    def __iter__ (self) :
        return chain(self._structure.inputs, self._structure.outputs)

    def inputs (self) :
        for node in self._structure.inputs :
            yield node.object

    def outputs (self) :
        for node in self._structure.outputs :
            yield node.object

    def scores (self) :
        for node in self._structure.outputs :
            yield Score(node.object, score=node.charge)

    def predict (self, input_objects, *args, **kw) :
        input_nodes = self._structure.inputs_for_objects(input_objects)

        for node in self._structure.feedforward(input_objects, *args, **kw) :
            yield Score(node.object, score=node.charge)

    def train (self, input_objects, output_objects, *args, **kw) :

        input_nodes  = self._structure.inputs_for_objects(input_objects)
        output_nodes = self._structure.outputs_for_objects(output_objects)

        return self._structure.backpropogate(input_nodes, output_nodes, *args, **kw)

    def forget (self) :
        ''' This resets the network to an untrained state. '''

        raise NotImplementedError # todo :

    def _associated (self, session) :
        return [self._structure]

class NeuralNetworkConfigurationError (exc.NLPLibError) :
    pass

class NeuralNetworkConfiguration (Base) :
    __slots__ = ('_layer_configurations',)

    def __init__ (self, *configs) :
        if len(configs) < 2 :
            raise NeuralNetworkConfigurationError('This requires a neural network that has at least two layers.')

        self._layer_configurations = list(self._make_layer_configurations(configs))

    def _make_layer_configurations (self, configs) :
        for config in configs :
            if isinstance(config, int) :
                yield Static(config)
            elif not isinstance(config, _LayerConfiguration) :
                yield StaticIO(config)
            else :
                yield config

    def __iter__ (self) :
        return iter(self._layer_configurations)

    def __getitem__ (self, index) :
        return self._layer_configurations[index]

    def hidden (self) :
        return self._layer_configurations[1:-1]

def random_weights (floor=-1.0, ceiling=1.0) :
    ''' This can be used to initialize the connections in the neural network with pseudorandom weights. '''

    while True :
        yield random.uniform(floor, ceiling)

class MakeStructure (Base) :

    def __init__ (self, structure, config, weights=random_weights) :
        self.structure = structure
        self.config = config
        self.weights = weights

    def __call__ (self) :
        self._connect(self._layers())

    def _make_layer (self, config) :
        layer = Layer(self.structure)

        if isinstance(config, StaticIO) :
            objects = [NeuralNetworkIO(layer, object) for object in config]
            layer.objects.extend(objects)
            size = len(objects)
        else :
            size = len(list(config))

        layer.values = (0.0,) * size
        layer.errors = (0.0,) * size

        return layer

    def _layers (self) :
        yield self._make_layer(self.config[0])

        for config in self.config[1:-1] :
            yield self._make_layer(config)

        yield self._make_layer(self.config[-1])

    def _connect (self, layers) :
        weights = self.weights()

        for input_layer, output_layer in paired(layers) :

            connection = Connection(self.structure)

            width  = len(input_layer)
            height = len(output_layer)

            connection.weights = tuple(tuple(next(weights, 0.0) for _ in range(width))
                                       for _ in range(height))

class Structure (Model) :
    def __init__ (self, config, **kw) :
        self.elements = []

        self.layers      = []
        self.connections = []

        if len(config) :
            MakeStructure(self, NeuralNetworkConfiguration(*config, **kw))()

    def __iter__ (self) :
        for (input_layer, output_layer), connection in zip(paired(self.layers), self.connections) :
            yield (input_layer, connection, output_layer)

    def __reversed__ (self) :
        return reversed(list(self))

    def _associated (self, session) :
        return self.elements

    def inputs_for_objects (self, objects) :
        # todo : could be made far more efficient
        for node in self.inputs :
            if node.object in objects :
                yield node

    def outputs_for_objects (self, objects) :
        # todo : could be made far more efficient
        for node in self.outputs :
            if node.object in objects :
                yield node

    def input (self) :
        return self.layers[0]

    def output (self) :
        return self.layers[-1]

    def hidden (self, reverse=False) :
        layers = self if not reverse else reversed(self)
        return truncated(itertools.islice(layers, 1, None), 1)

    def paired (self, reverse=False) :
        layers = self if not reverse else reversed(self)
        return paired(layers)

    def feedforward (self, input_nodes, *args, **kw) :
        return Feedforward(self, input_nodes, *args, **kw)()

    def backpropogate (self, input_nodes, output_nodes, rate=0.2, activation_derivative=math.dtanh,
                       **kw) :
        ''' This method allows for supervised training, using the backpropagation algorithm. '''

        self.feedforward(input_nodes, **kw)

        return Backpropagate(self, input_nodes, output_nodes, rate=rate, activation_derivative=activation_derivative)()

    def clear (self) :
        ''' This deletes all of the network's nodes, leaving an empty network. '''

        raise NotImplementedError # todo :

class Element (Model) :
    ''' This is a base class structural neural network elements, e.g., links, nodes, or IO nodes. '''

    def __init__ (self, structure) :
        self.structure = structure

class Layer (Element) :
    def __init__ (self, structure) :
        super().__init__(structure)

        self.values = ()
        self.errors = ()

        self.objects = []

    def __repr__ (self, *args, **kw) :
        return super().__repr__(len(self.values), *args, **kw)

    def __iter__ (self) :
        return iter(self.values)

    def __len__ (self) :
        return len(self.values)

class Connection (Element) :
    def __init__ (self, structure) :
        super().__init__(structure)

        self.weights = ()

    def __repr__ (self, *args, **kw) :
        try :
            width = len(self.weights[0])
        except IndexError :
            width = 0

        height = len(self.weights)

        dimensions = type('Dimensions', (), {'__repr__' : lambda self : '{}x{}'.format(width, height)})()

        return super().__repr__(dimensions, *args, **kw)

class NeuralNetworkIO (Element) :
    def __init__ (self, layer, object) :
        super().__init__(layer.structure)

        self.layer = layer
        self.object = object

    def __repr__ (self, *arg, **kw) :
        return super().__repr__(self.object, *arg, **kw)

    def _make_object (self) :
        if self._model is not None :
            self._object = self._model
        else :
            self._object = self._pickled

    @property
    def object (self) :
        return self._object

    @object.setter
    def object (self, value) :
        self._model = None
        self._pickled = None

        if isinstance(value, Seq) : # todo : make handle all models not just Seq
            self._model = value
        else :
            self._pickled = value

        self._object = value

