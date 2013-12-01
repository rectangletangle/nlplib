

from nlplib.core.model.base import Model
from nlplib.general.iter import windowed, chop

# todo : implement __all__ list

class LayerConfiguration (Model) :
    def __call__ (self, layer) :
        pass

def DynamicLayer (LayerConfiguration) :
    pass

class StaticLayer (LayerConfiguration) :
    def __init__ (self, size) :
        self.size = int(size)

    def __repr__ (self) :
        return super().__repr__(size=self.size)

    def __call__ (self, layer) :
        for _ in range(self.size) :
            layer.add()

class StaticIOLayer (LayerConfiguration) :
    def __init__ (self, seqs) :
        self.seqs = list(seqs)

    def __repr__ (self) :
        seq_count = len(self.seqs)
        if seq_count > 5 :
            return super().__repr__(size=seq_count)
        else :
            return super().__repr__(seqs=self.seqs)

    def __call__ (self, layer) :
        for seq in self.seqs :
            layer.add(seq)

class NeuralNetwork (Model) :
    def __init__ (self, name, elements=(), links=(), nodes=(), io_nodes=()) :
        self.name = name

        self.elements = list(elements)
        self.links    = list(links)
        self.nodes    = list(nodes)
        self.io_nodes = list(io_nodes)

    def __repr__ (self, *args, **kw) :
        return super().__repr__(self.name, *args, **kw)

class LayeredNeuralNetwork (NeuralNetwork) :
    def __init__ (self, name, layer_configurations=(), *args, **kw) :
        super().__init__(name, *args, **kw)

        self.layer_configurations = list(layer_configurations)

    def __repr__ (self) :
        if len(self.layer_configurations) > 5 :
            return super().__repr__()
        else :
            return super().__repr__(self.layer_configurations)

    def __len__ (self) :
        return len(self.layer_configurations)

    def input_layer_index (self) :
        return 0

    def hidden_layer_indexes (self) :
        return range(self.input_layer_index() + 1, self.output_layer_index())

    def output_layer_index (self) :
        return len(self) - 1

    def input_layer_configuration (self) :
        return self.layer_configurations[self.input_layer_index()]

    def hidden_layer_configurations (self) :
        indexes = self.hidden_layer_indexes()
        return self.layer_configurations[indexes[0]:indexes[-1]]

    def output_layer_configuration (self) :
        return self.layer_configurations[self.output_layer_index()]

    def paired_layer_configurations (self) :
        size = 2
        return chop(windowed(self, size), size)

class NeuralNetworkElement (Model) :
    def __init__ (self, neural_network) :
        self.neural_network_id = neural_network.id

class Link (NeuralNetworkElement) :
    ''' A class which links together neural network nodes. '''

    def __init__ (self, neural_network, input_node, output_node, strength=1.0) :
        super().__init__(neural_network)
        self.input_node_id  = input_node.id
        self.output_node_id = output_node.id

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
        self.seq_id = seq.id

