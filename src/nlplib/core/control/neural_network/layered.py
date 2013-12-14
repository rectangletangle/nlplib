

from nlplib.core.control.neural_network.base import NeuralNetworkDependent
from nlplib.core.model import Link, Node, IONode
from nlplib.general.iter import windowed, chop
from nlplib.core import Base

# todo : implement __all__ list

def static (size) :
    size = int(size)

    def make (layer) :
        for _ in range(size) :
            layer.add()

    return make

def static_io (seqs) :
    def make (layer) :
        for seq in seqs :
            layer.add(seq)

    return make

def dynamic () :
    raise NotImplementedError

class Layer (NeuralNetworkDependent) :
    def __init__ (self, session, neural_network, layer_index) :
        super().__init__(session, neural_network)

        self.layer_index = layer_index

        self.nodes = []

    def __iter__ (self) :
        return iter(self.nodes)

    def __len__ (self) :
        return len(self.nodes)

    def add (self) :
        node = self.session.add(Node(self.neural_network, layer_index=self.layer_index))
        self.nodes.append(node)
        return node

class IOLayer (Layer) :
    def __init__ (self, session, neural_network, layer_index, is_input, *args, **kw) :
        super().__init__(session, neural_network, layer_index, *args, **kw)
        self.is_input = is_input

    def add (self, seq) :
        io_node = self.session.add(IONode(self.neural_network, seq, self.is_input, layer_index=self.layer_index))
        self.nodes.append(io_node)
        return io_node

class _LayeredStructure (Base) :
    def __init__ (self, layer_config) :
        self.layer_config = list(layer_config)

    def __len__ (self) :
        return len(self.layer_config)

    def input_layer_index (self) :
        return 0

    def hidden_layer_indexes (self) :
        return range(self.input_layer_index() + 1, self.output_layer_index())

    def output_layer_index (self) :
        return len(self) - 1

    def input_layer_configuration (self) :
        return self.layer_config[self.input_layer_index()]

    def hidden_layer_configurations (self) :
        indexes = self.hidden_layer_indexes()
        return self.layer_config[indexes[0]:indexes[-1]]

    def output_layer_configuration (self) :
        return self.layer_config[self.output_layer_index()]

class MakeLayeredNeuralNetwork (NeuralNetworkDependent) :
    def __init__ (self, session, neural_network, layer_config) :
        super().__init__(session, neural_network)
        self.structure = _LayeredStructure(layer_config)

    def layers (self) :
        yield IOLayer(self.session, self.neural_network, self.structure.input_layer_index(), True)

        for layer_index in self.structure.hidden_layer_indexes() :
            yield Layer(self.session, self.neural_network, layer_index)

        yield IOLayer(self.session, self.neural_network, self.structure.output_layer_index(), False)

    def add_nodes (self, layers) :
        for config, layer in zip(self.structure.layer_config, layers) :
            if not callable(config) :
                config = static_layer(int(config))

            config(layer)

            yield layer

    def link_up (self, layers) :
        size = 2
        for input_layer, output_layer in chop(windowed(layers, size=size, step=1), size) :
            for input_node in input_layer :
                for output_node in output_layer :
                    self.session.add(Link(self.neural_network, input_node, output_node, strength=0.2))

    def __call__ (self) :
        self.link_up(self.add_nodes(self.layers()))

def _test_layered_structure (ut) :
    structure = _LayeredStructure((static(1), 1))

    ut.assert_equal(list(structure.hidden_layer_indexes()), [])

    structure = _LayeredStructure(static(1) for _ in range(3))

    ut.assert_equal(list(structure.hidden_layer_indexes()), [1] )
    ut.assert_equal(structure.output_layer_index(),         2   )
    ut.assert_equal(structure.input_layer_index(),          0   )

    structure = _LayeredStructure(static(2) for _ in range(4))

    ut.assert_equal(list(structure.hidden_layer_indexes()), [1, 2] )
    ut.assert_equal(structure.output_layer_index(),         3      )
    ut.assert_equal(structure.input_layer_index(),          0      )

def __test__ (ut) :
    _test_layered_structure(ut)

if __name__ == '__main__' :
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())

