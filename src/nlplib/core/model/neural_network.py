

from nlplib.core.model.base import Model

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

class NeuralNetworkElement (Model) :
    def __init__ (self, neural_network) :
        self.neural_network = neural_network

class Link (NeuralNetworkElement) :
    ''' A class which links together neural network nodes. '''

    def __init__ (self, neural_network, input_node, output_node, strength=1.0) :
        super().__init__(neural_network)

        self.input_node  = input_node
        self.output_node = output_node

        self.strength = strength

class Node (NeuralNetworkElement) :
    ''' A class for neural network nodes. '''

    def __init__ (self, neural_network, layer_index=None, current=1.0, input_nodes=(), output_nodes=()) :
        super().__init__(neural_network)

        self.layer_index = layer_index
        self.current = current

        self.input_nodes  = list(input_nodes)
        self.output_nodes = list(output_nodes)

class IONode (Node) :
    ''' A class for input and output neural network nodes. These are the nodes on the edge of a neural network, which
        hold data (in this case sequences). '''

    def __init__ (self, neural_network, seq, *args, **kw) :
        super().__init__(neural_network, *args, **kw)

        self.seq = seq

