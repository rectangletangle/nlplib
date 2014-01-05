

import itertools
import pickle

from nlplib.core.model.naturallanguage import Seq
from nlplib.core.model.neuralnetwork.alg import Feedforward, Backpropagate
from nlplib.core.model.base import Model
from nlplib.general.represent import pretty_float
from nlplib.general.iterate import truncated, paired
from nlplib.general import math

__all__ = ['NeuralNetwork', 'Perceptron', 'NeuralNetworkElement', 'Link', 'Node', 'IONode']

class NeuralNetwork (Model) :
    def __init__ (self, name) :

        self.name = name

# todo : move to structure
#############################################
        self.elements = []
        self.inputs   = []
        self.outputs  = []

    def _iter_nodes (self, from_nodes, direction) :
        from_nodes = set(from_nodes)
        yield set(from_nodes) # Returning a copy is desirable here.

        while True :
            to_nodes = {to_node
                        for from_node in from_nodes
                        for to_node in direction(from_node)}

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

    def hidden (self, reverse=False) :
        layers = self if not reverse else reversed(self)
        return truncated(itertools.islice(layers, 1, None), 1)

    def paired (self, reverse=False) :
        layers = self if not reverse else reversed(self)
        return paired(layers)
#############################################

    def __repr__ (self, *args, **kw) :
        return super().__repr__(self.name, *args, **kw)

    def feedforward (self, active_input_nodes, *args, **kw) : # todo : use input objects
        return Feedforward(self, active_input_nodes, *args, **kw)()

    def backpropogate (self, active_input_nodes, correct_output_nodes, rate=0.2, activation_derivative=math.dtanh,
                       **kw) :
        ''' This method allows for supervised training, using the backpropagation algorithm. '''

        self.feedforward(active_input_nodes, **kw)

        return Backpropagate(self, active_input_nodes, correct_output_nodes, rate=rate, # todo : use input, and correct objects
                             activation_derivative=activation_derivative)()

##    def __contains__ (self) :
##        return bool() in object in inputs or outputs
##
##    def __iter__ (self) :
##        return chain(self.inputs(), self.outputs())
##
##    def inputs (self) :
##        return iter(input objects)
##
##    def outputs (self) :
##        return iter(output objects)
##
##    def scores (self) :
##        yield last scores from output layer
##
##    def predict (self) :
##        return scores from structure.feedforward
##
##    def train (self) :
##        return error from structure.backpropogate
##
##   def forget (self) :
##        ''' This resets the network to an untrained state. '''
##
##       raise NotImplementedError # todo :

class NNStructure :
    def __init__ (self) :
        numpy or nlplib

    def feedforward (self) :
        pass

    def backpropogate (self) :
        pass

    def clear (self) :
        ''' This deletes all of the network's nodes, leaving an empty network. '''

        raise NotImplementedError # todo :

class Perceptron (NeuralNetwork) :
    # todo : use numpy_
    pass

class NeuralNetworkElement (Model) :
    ''' This is a base class neural network elements, e.g., links, nodes, or IO nodes. '''

    def __init__ (self, neural_network) :
        self.neural_network = neural_network

class Link (NeuralNetworkElement) :
    ''' This class is used for linking together neural network nodes. The affinity attribute denotes how strong the
        connection between the nodes is. '''

    def __init__ (self, neural_network, input_node, output_node, affinity=1.0) :

        super().__init__(neural_network)

        self.input_node  = input_node
        self.output_node = output_node

        self.affinity = affinity

    def __repr__ (self, *arg, **kw) :
        return super().__repr__(pretty_float(self.affinity), self.input_node, self.output_node, *arg, **kw)

class Node (NeuralNetworkElement) :
    ''' A class for neural network nodes. The charge attribute designates how "excited" the node is, at a given
        moment. '''

    def __init__ (self, neural_network, charge=0.0, error=None) :

        super().__init__(neural_network)

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

    def __init__ (self, neural_network, object, is_input, other=None, *args, **kw) :

        super().__init__(neural_network, *args, **kw)

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

