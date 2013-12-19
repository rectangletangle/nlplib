

import itertools

from nlplib.core.model.base import Model
from nlplib.general.iter import truncate
from nlplib.general import pretty_float

# todo : implement __all__ list

class NeuralNetwork (Model) :
    def __init__ (self, name, elements=(), links=(), nodes=(), io_nodes=()) :
        self.name = name

        self.elements = list(elements)
        self.links    = list(links)
        self.nodes    = list(nodes)
        self.io_nodes = list(io_nodes)

    def __repr__ (self, *args, **kw) :
        return super().__repr__(self.name, *args, **kw)

    def _iter_layers (self, from_nodes, direction) :
        yield set(from_nodes)

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
        return self._iter_layers(self.input(), lambda input_node : input_node.output_nodes)

    def __reversed__ (self) :
        return self._iter_layers(self.output(), lambda output_node : output_node.input_nodes)

    def input (self) :
        for io_node in self.io_nodes :
            if io_node.is_input :
                yield io_node

    def output (self) :
        for io_node in self.io_nodes :
            if not io_node.is_input :
                yield io_node

    def hidden (self, reverse=False) :
        layers = self if not reverse else reversed(self)
        return truncate(itertools.islice(layers, 1, None), 1)

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

    def __init__ (self, neural_network, charge=1.0, error=None) :
        super().__init__(neural_network)

        self.charge = charge
        self.error  = error

        self.input_nodes  = {}
        self.output_nodes = {}

    def __repr__ (self, *arg, **kw) :
        return super().__repr__(pretty_float(self.charge), *arg, **kw)

class IONode (Node) :
    ''' A class for input and output neural network nodes. These are the nodes on the edge of a neural network, which
        hold data (in this case sequences). '''

    def __init__ (self, neural_network, seq, is_input, *args, **kw) :
        super().__init__(neural_network, *args, **kw)

        self.seq = seq

        self.is_input = bool(is_input)

    def __repr__ (self, *arg, **kw) :
        return super().__repr__(self.seq, *arg, **kw)

