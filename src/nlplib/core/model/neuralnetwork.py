

import itertools
import pickle
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

__all__ = ['NeuralNetwork', 'Link', 'Node', 'IONode']

class NeuralNetwork (Model) :
    def __init__ (self, *config, name=None, **kw) :
        self.name = name

        self._structure = Structure(config, **kw)

    def _associated (self, session) :
        return [self._structure]

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

def random_affinity (floor=-1.0, ceiling=1.0) :
    ''' This can be used to initialize the links in the neural network with pseudorandom affinities (weights). '''

    while True :
        yield random.uniform(floor, ceiling)

class MakeMultiLayer (Base) :

    def __init__ (self, structure, config, affinity=random_affinity) :
        self.structure = structure
        self.config = config
        self.affinity = affinity

    def __call__ (self) :
        for layers in self._link_up(self._layers()) :
            pass

    def __iter__ (self) :
        return united(self._link_up(self._layers()))

    def _layers (self) :
        yield [IONode(self.structure, object, is_input=True) for object in self.config[0]]

        for config in self.config[1:-1] :
            yield [Node(self.structure) for _ in config]

        yield [IONode(self.structure, object, is_input=False) for object in self.config[-1]]

    def _link_up (self, layers) :
        affinity = self.affinity()

        for input_layer, output_layer in paired(layers) :
            for input_node in input_layer :
                for output_node in output_layer :
                    link = Link(self.structure, input_node, output_node, affinity=next(affinity, 0.0))

                    input_node.outputs[output_node] = link
                    output_node.inputs[input_node]  = link

            yield (input_layer, output_layer)

class Structure :
    def __init__ (self, config, **kw) :
        self.elements = []
        self.inputs   = []
        self.outputs  = []

        if len(config) :
            MakeMultiLayer(self, NeuralNetworkConfiguration(*config, **kw))()

    def _iter_nodes (self, from_nodes, direction) :
        from_nodes = set(from_nodes)
        yield {node for node in from_nodes if node is not None}

        while True :
            to_nodes = {to_node
                        for from_node in from_nodes if from_node is not None
                        for to_node in direction(from_node) if to_node is not None}

            if not len(to_nodes) :
                break
            else :
                yield set(to_nodes)
                from_nodes = to_nodes

    def __iter__ (self) :
        return self._iter_nodes(self.inputs, lambda input_node : input_node.outputs)

    def __reversed__ (self) :
        return self._iter_nodes(self.outputs, lambda output_node : output_node.inputs)

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

        self._charges_string = ''
        self._errors_string  = ''

    @property
    def charges () :
        for charge in self._charges.split() :
            yield float(charge)

    @property
    def errors () :
        for error in self._errors.split() :
            yield float(error)

    def __iter__ (self) :
        return iter(self.charges)

    def __len__ (self) :
        return len(self.charges)

class Link (Element) :
    ''' This class is used for linking together neural network nodes. The affinity attribute denotes how strong the
        connection between the nodes is. '''

    def __init__ (self, structure, input_node, output_node, affinity=1.0) :

        super().__init__(structure)

        self.input_node  = input_node
        self.output_node = output_node

        self.affinity = affinity

    def __repr__ (self, *arg, **kw) :
        return super().__repr__(pretty_float(self.affinity), self.input_node, self.output_node, *arg, **kw)

class Node (Element) :
    ''' A class for neural network nodes. The charge attribute designates how "excited" the node is, at a given
        moment. '''

    def __init__ (self, structure, charge=0.0, error=None) :

        super().__init__(structure)

        self.charge = charge
        self.error  = error

        self.inputs  = {}
        self.outputs = {}

    def __repr__ (self, *arg, **kw) :
        return super().__repr__(pretty_float(self.charge), *arg, **kw)

    def _associated (self, session) :
        yield from self.inputs.values()
        yield from self.outputs.values()

class IONode (Node) :
    ''' This is a class for input and output neural network nodes. These are the nodes on the edges of the network
        which communicate with the outside world. Nodes store their meaning with their object property. Although the
        object property is normally used for holding natural language sequences (e.g. words, grams), <None> or any
        other pickle-able Python object can be used. '''

    def __init__ (self, structure, object, is_input, other=None, *args, **kw) :

        super().__init__(structure, *args, **kw)

        self.object = object

        self.is_input = bool(is_input)

    def _serialize (self, *args, **kw) :
        return pickle.dumps(*args, **kw)

    def _deserialize (self, *args, **kw) :
        return pickle.loads(*args, **kw)

    def __repr__ (self, *arg, **kw) :
        return super().__repr__(self.object, *arg, **kw)

    def _make_object (self) :
        if self._model is not None :
            self._object = self._model
        else :
            self._object = self._deserialize(self._serialized_object)

    @property
    def object (self) :
        return self._object

    @object.setter
    def object (self, value) :
        self._model = None
        self._serialized_object = None

        if isinstance(value, Seq) : # todo : make handle all models not just Seq
            self._model = value
        else :
            self._serialized_object = self._serialize(value)

        self._object = value

