

import itertools

from nlplib.core.model.base import Model
from nlplib.general.represent import pretty_float
from nlplib.general.iterate import truncated, paired
from nlplib.general import composite

# todo : implement __all__ list

class NeuralNetwork (Model) :
    def __init__ (self, name) :
        self.name = name

        self.elements = []
        self.inputs   = []
        self.outputs  = []

    def __repr__ (self, *args, **kw) :
        return super().__repr__(self.name, *args, **kw)

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

    def __associated__ (self) :
        return self.elements

    def hidden (self, reverse=False) :
        layers = self if not reverse else reversed(self)
        return truncated(itertools.islice(layers, 1, None), 1)

    def paired (self, reverse=False) :
        layers = self if not reverse else reversed(self)
        return paired(layers)

class MLPNeuralNetwork (NeuralNetwork) :
    pass

class NeuralNetworkElement (Model) :

    def __init__ (self, neural_network) :
        self.neural_network = neural_network

class Link (NeuralNetworkElement) :
    ''' A class which links together neural network nodes. '''

    def __init__ (self, neural_network, input_node, output_node, affinity=1.0) :
        super().__init__(neural_network)

        self.input_node  = input_node
        self.output_node = output_node

        self.affinity = affinity

    def __repr__ (self, *arg, **kw) :
        return super().__repr__(pretty_float(self.affinity), self.input_node, self.output_node, *arg, **kw)

class Node (NeuralNetworkElement) :
    ''' A class for neural network nodes. '''

    def __init__ (self, neural_network, charge=0.0, error=None) :
        super().__init__(neural_network)

        self.charge = charge
        self.error  = error

        self.inputs  = {}
        self.outputs = {}

    def __repr__ (self, *arg, **kw) :
        return super().__repr__(pretty_float(self.charge), *arg, **kw)

    def __associated__ (self) :
        yield from self.inputs.values()
        yield from self.outputs.values()

class IONode (Node) :
    ''' A class for input and output neural network nodes. These are the nodes on the edge of a neural network, which
        hold data (in this case sequences). '''

    def __init__ (self, neural_network, seq, is_input, *args, **kw) :
        super().__init__(neural_network, *args, **kw)

        self.seq = seq

        self.is_input = bool(is_input)

    def __repr__ (self, *arg, **kw) :
        return super().__repr__(self.seq, *arg, **kw)

