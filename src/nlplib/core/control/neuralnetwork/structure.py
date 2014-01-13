

import random

import nlplib.core.model

from nlplib.general.iterate import paired
from nlplib.general.represent import pretty_ellipsis
from nlplib.core.base import Base
from nlplib.core import exc

try :
    from nlplib.core.control.neuralnetwork.numpy_ import Array, Matrix
except ImportError :
    from nlplib.core.control.neuralnetwork import Array, Matrix

__all__ = ['NeuralNetworkConfigurationError', 'random_weights', 'LayerConfiguration', 'StaticLayer', 'StaticIOLayer',
           'Dynamic', 'DynamicIO', 'NeuralNetworkConfiguration', 'MakeStructure']

class NeuralNetworkConfigurationError (exc.NLPLibError) :
    pass

def random_weights (floor=-1.0, ceiling=1.0) :
    ''' This can be used to initialize the connections in the neural network with pseudorandom weights. '''

    while True :
        yield random.uniform(floor, ceiling)

class LayerConfiguration (Base) :
    pass

class StaticLayer (LayerConfiguration) :
    def __init__ (self, size) :
        self.size = size

    def __repr__ (self, *args, **kw) :
        return super().__repr__(self.size, *args, **kw)

    def __iter__ (self) :
        for _ in range(self.size) :
            yield None

    def __len__ (self) :
        return self.size

class StaticIOLayer (LayerConfiguration) :
    def __init__ (self, objects) :
        self.objects = list(objects)

    def __repr__ (self, *args, **kw) :
        limit = 5

        if len(self.objects) > limit :
            pretty_objects = tuple(self.objects[:limit] + [pretty_ellipsis()])
        else :
            pretty_objects = tuple(self.objects)

        return super().__repr__(*pretty_objects + args, **kw)

    def __iter__ (self) :
        return iter(self.objects)

    def __len__ (self) :
        return len(self.objects)

class Dynamic (LayerConfiguration) :
    def __init__ (self, *args, **kw) :
        raise NotImplementedError

class DynamicIO (LayerConfiguration) :
    def __init__ (self, *args, **kw) :
        raise NotImplementedError

class NeuralNetworkConfiguration (Base) :
    __slots__ = ('_layer_configurations',)

    def __init__ (self, *configs) :
        if len(configs) < 2 :
            raise NeuralNetworkConfigurationError('The neural network must have at least two layers.')

        self._layer_configurations = list(self._make_layer_configurations(configs))

    def _make_layer_configurations (self, configs) :
        for config in configs :
            if isinstance(config, int) :
                yield StaticLayer(config)
            elif not isinstance(config, LayerConfiguration) :
                yield StaticIOLayer(config)
            else :
                yield config

    def __iter__ (self) :
        return iter(self._layer_configurations)

    def __getitem__ (self, index) :
        return self._layer_configurations[index]

    def hidden (self) :
        return self._layer_configurations[1:-1]

class MakeStructure (Base) :

    def __init__ (self, structure, config, weights) :
        self.structure = structure
        self.config = NeuralNetworkConfiguration(*config)
        self.weights = weights

    def __call__ (self) :
        self._connect(self._layers())

    def _make_layer (self, config) :
        layer = nlplib.core.model.Layer(self.structure)

        if isinstance(config, StaticIOLayer) :
            ios = [nlplib.core.model.NeuralNetworkIO(self.structure, object) for object in config]
            layer.io.extend(ios)
            size = len(ios)
        else :
            size = len(list(config))

        layer._charges = Array([0.0] * size)
        layer._errors  = Array([0.0] * size)

        return layer

    def _layers (self) :
        yield self._make_layer(self.config[0])

        for config in self.config[1:-1] :
            yield self._make_layer(config)

        yield self._make_layer(self.config[-1])

    def _initial_weights (self) :
        if callable(self.weights) :
            return self.weights()
        else :
            return (self.weights for _ in itertools.count())

    def _connect (self, layers) :
        weights = self._initial_weights()

        for input_layer, output_layer in paired(layers) :

            connection = nlplib.core.model.Connection(self.structure)

            width  = len(input_layer)
            height = len(output_layer)

            connection._weights = Matrix(tuple(next(weights, 0.0) for _ in range(width))
                                         for _ in range(height))

