# todo : a lot

from math import tanh

from nlplib.core.model import Seq, Link, Node, IONode, Word, NeuralNetworkElement, SessionDependent

from nlplib.general.iter import windowed, chop
from nlplib.core import Base

##def dtanh (y) :
##    return 1.0 - y * y
##
##def access_nodes_in_layer (session, neural_network, layer) :
##    return session._sqlalchemy_session.query(Node).filter_by(neural_network_id=neural_network.id, layer=layer).all()
##
##def input_nodes_for_seqs (session, seqs) :
##    return [session._sqlalchemy_session.query(IONode).filter_by(layer=0, seq_id=seq.id).one() for seq in seqs]
##
##
##
##class NodeAccess (SessionDependent) :
##    # todo : merge with Access when done.
##
##    def input_nodes (self, structure) :
##        return nodes_at(self.session, structure.input_node_layer())
##
##    def hidden_nodes (self, structure) :
##        for layer in structure.hidden_node_layers() :
##            yield nodes_at(self.session, layer)
##
##    def output_nodes (self, structure) :
##        return nodes_at(self.session, structure.output_node_layer())
##
##    def io_paired_nodes (self, structure) :
##        for input_layer, output_layer in structure.io_paired_layers() :
##            # This could be done with far fewer <nodes_at> queries.
##            yield (nodes_at(self.session, input_layer), nodes_at(self.session, output_layer))
##
##
##
##class BackPropagate (SessionDependent) :
##    def back_propagate(self, targets, n=0.5) :
##        input_nodes  = self.nodes(0)
##        hidden_nodes = self.nodes(1)
##        output_nodes = self.nodes(2)
##
##        output_deltas = [0.0] * len(targets)
##        for i, output_node in enumerate(output_nodes) :
##            error = targets[i] - output_node.current
##            output_deltas[i] = dtanh(output_node.current) * error
##
##        hidden_deltas = [0.0] * len(hidden_nodes)
##        for i, hidden_node in enumerate(hidden_nodes) :
##            error = sum(output_deltas[j] * self._get_link(hidden_node, output_node).strength
##                        for j, output_node in enumerate(output_nodes))
##            hidden_deltas[i] = dtanh(hidden_node.current) * error
##
##        for i, hidden_node in enumerate(hidden_nodes) :
##            for j, output_node in enumerate(output_nodes) :
##                change = output_deltas[j] * hidden_node.current
##                self._get_link(hidden_node, output_node).strength += n * change
##
##        for i, input_node in enumerate(input_nodes) :
##            for j, hidden_node in enumerate(hidden_nodes) :
##                change = hidden_deltas[j] * input_node.current
##                try :
##                    self._get_link(input_node, hidden_node).strength += n * change
##                except :
##
##                    print([node.id for node in input_node.output_nodes],
##                          [node.id for node in hidden_node.input_nodes])
##                    raise
##
##    def __call__ (self, input_seqs, output_seqs, correct) :
##
##        #self.build(input_seqs, output_seqs)
##
##        self.input(input_seqs)
##
##        targets = [0.0] * len(output_seqs)
##
##        targets[output_seqs.index(correct)] = 1.0
##
##        error = self.back_propagate(targets)
##
##    def input (self, seqs) :
##        # redundant
##        return self.feed_forward(input_nodes_for_seqs(self.session, seqs))

class NeuralNetworkDependent (SessionDependent) :
    def __init__ (self, session, neural_network, *args, **kw) :
        super().__init__(session, *args, **kw)
        self.neural_network = neural_network

class Layer (SessionDependent) :
    def __init__ (self, session, neural_network, layer_index) :
        super().__init__(session)

        self.neural_network = neural_network
        self.layer_index    = layer_index

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
    def add (self, seq) :
        io_node = self.session.add(IONode(self.neural_network, seq, layer_index=self.layer_index))
        self.nodes.append(io_node)
        return

class MakeLayeredNeuralNetwork (NeuralNetworkDependent) :
    def link_up (self, layers) :
        size = 2
        for input_layer, output_layer in chop(windowed(layers, size=size, step=1), size) :
            for input_node in input_layer :
                for output_node in output_layer :
                    self.session.add(Link(self.neural_network, input_node, output_node, strength=0.2))

    def make_layers (self) :
        yield IOLayer(self.session, self.neural_network, self.neural_network.input_layer_index())

        for layer_index in self.neural_network.hidden_layer_indexes() :
            yield Layer(self.session, self.neural_network, layer_index)

        yield IOLayer(self.session, self.neural_network, self.neural_network.output_layer_index())

    def add_nodes (self, layers) :
        for configure, layer in zip(self.neural_network.layer_configurations, layers) :
            if not callable(configure) :
                configure = StaticLayer(int(configure))

            configure(layer)

            yield layer

    def __call__ (self) :
        self.link_up(self.add_nodes(self.make_layers()))

class FeedForward (NeuralNetworkDependent) :
    def _access_link (self, input_node, output_node) :
        return self.session.access.link(self.neural_network, input_node, output_node)

    def activate (self, input_nodes) :
        for input_node in input_nodes :
            input_node.current = 1.0

        return input_nodes

    def fire (self, input_nodes) :

        output_nodes = {output_node
                        for input_node in input_nodes
                        for output_node in input_node.output_nodes}

        for output_node in output_nodes :
            total = sum(input_node.current * self._access_link(input_node, output_node).strength
                        for input_node in input_nodes)
            output_node.current = tanh(total)

        return output_nodes

    def __call__ (self, active_input_nodes) :
        active_input_nodes = list(active_input_nodes)

        active_nodes = self.activate(active_input_nodes)

        while True :
            current_nodes = self.fire(active_nodes)

            if not len(current_nodes) :
                break
            else :
                active_nodes = current_nodes

        return active_nodes

def ask (session, neural_network, seqs) :
    active_input_nodes = session.access.nodes_for_seqs(neural_network, seqs, layer_index=0)

    return [(node, node.current) for node in FeedForward(session, neural_network)(active_input_nodes)]

def __demo__ () :
    from nlplib.core.model import Database, LayeredNeuralNetwork, StaticIOLayer, StaticLayer

    db = Database()

    with db as session :
        for char in 'abcdef' :
            session.add(Word(char))

    with db as session :
        config = (StaticIOLayer(session.access.words('a b')),
                  StaticLayer(1),
                  StaticIOLayer(session.access.words('d e f')))

        session.add(LayeredNeuralNetwork('foo', config))

    from pprint import pprint

    with db as session :
        MakeLayeredNeuralNetwork(session, session.access.neural_network('foo'))()

    with db as session :
        print(ask(session, session.access.neural_network('foo'), session.access.words('a b')))

def _test_structure (ut) :
    raise DeprecationWarning
    structure = Structure((static(1), 1))
    ut.assert_equal(list(structure.io_paired_layers()),   [(0, 1)] )
    ut.assert_equal(list(structure.all_layers()),         [0, 1]   )
    ut.assert_equal(list(structure.hidden_node_layers()), []       )

    structure = Structure(static(1) for _ in range(3))
    ut.assert_equal(list(structure.io_paired_layers()),   [(0, 1), (1, 2)] )
    ut.assert_equal(list(structure.all_layers()),         [0, 1, 2]        )
    ut.assert_equal(list(structure.hidden_node_layers()), [1]              )
    ut.assert_equal(structure.output_node_layer(),        2                )
    ut.assert_equal(structure.input_node_layer(),         0                )

    structure = Structure(static(2) for _ in range(4))
    ut.assert_equal(list(structure.io_paired_layers()),   [(0, 1), (1, 2), (2, 3)] )
    ut.assert_equal(list(structure.all_layers()),         [0, 1, 2, 3]             )
    ut.assert_equal(list(structure.hidden_node_layers()), [1, 2]                   )
    ut.assert_equal(structure.output_node_layer(),        3                        )
    ut.assert_equal(structure.input_node_layer(),         0                        )

if __name__ == '__main__' :
    __demo__()

