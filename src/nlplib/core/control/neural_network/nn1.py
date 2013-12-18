

import itertools
import random

from nlplib.general import math
from nlplib.general.iter import chunked, chop, windowed

class Link :
    def __init__ (self, input_node, output_node, affinity) :
        self.input_node = input_node
        self.output_node = output_node
        self.affinity = affinity

class Node :
    i = -1
    def __init__ (self, charge) :
        Node.i += 1
        self.i = Node.i

        self.charge = charge

        self.error = None

        self.input_nodes  = {}
        self.output_nodes = {}

    def __repr__ (self) :
        return '<%s %d %f>' % (self.__class__.__name__, self.i, self.charge)

class IONode (Node) :
    def __init__ (self, neural_network, charge, is_input) :
        self.neural_network = neural_network
        self.neural_network.io_nodes.append(self)
        self.is_input = is_input

        super().__init__(charge)

class NeuralNetwork :
    def __init__ (self) :
        self.io_nodes = []

    def input_nodes (self) :
        return [io_node for io_node in self.io_nodes if io_node.is_input]

    def output_nodes (self) :
        return [io_node for io_node in self.io_nodes if not io_node.is_input]

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
        return self._iter_layers(self.input_nodes(), lambda input_node : input_node.output_nodes)

    def __reversed__ (self) :
        return self._iter_layers(self.output_nodes(), lambda output_node : output_node.input_nodes)

class Make :
    def __init__ (self, neural_network, config) :
        self.neural_network = neural_network
        self.config = config

    def __call__ (self) :
        self.link_up(self.layers())

    def layers (self) :
        charge = 0.0
        yield [IONode(self.neural_network, charge, True) for _ in range(self.config[0])]
        for count in self.config[1:-1] :
            yield [Node(charge) for _ in range(count)]
        yield [IONode(self.neural_network, charge, False) for _ in range(self.config[-1])]

    def link_up (self, layers) :
        for input_layer, output_layer in chop(windowed(layers, 2, 1), 2) :
            for input_node in input_layer :
                for output_node in output_layer :
                    link = Link(input_node, output_node, random.randint(-1, 1))
                    input_node.output_nodes[output_node] = link
                    output_node.input_nodes[input_node] = link

class FeedForward :
    def __init__ (self, neural_network, active_input_nodes) :
        self.neural_network = neural_network
        self.active_input_nodes = active_input_nodes

    def input_strength (self, node) :
        return sum(link.input_node.charge * link.affinity for link in node.input_nodes.values())

    def __iter__ (self) :
        for node in self.neural_network.input_nodes() :
            node.charge = 0.0

        for node in self.active_input_nodes :
            node.charge = 1.0

        for layer in itertools.islice(self.neural_network, 1, None) :
            for node in layer :
                node.charge = math.tanh(self.input_strength(node))

        yield from layer

class Backpropagate :
    def __init__ (self, neural_network, correct, rate=0.01) :
        self.neural_network = neural_network
        self.correct = correct
        self.rate = rate

    def __call__ (self) :
        for value, output_node in zip(self.correct, self.neural_network.output_nodes()) :
            output_node.error = math.dtanh(output_node.charge) * (value - output_node.charge)

        for layer in list(reversed(self.neural_network))[1:-1] :
            for node in layer :
                node.error = math.dtanh(node.charge) * sum(output_node.error * link.affinity
                                                           for output_node, link in node.output_nodes.items())

        for layer in list(reversed(self.neural_network))[1:] :
            for node in layer :
                for link in node.output_nodes.values() :
                    link.affinity += self.rate * link.output_node.error * node.charge

        return sum(0.5 * (value - node.charge) ** 2
                   for value, node in zip(self.correct, self.neural_network.output_nodes()))

class NN :
    def __init__ (self, config) :
        self.neural_network = NeuralNetwork()
        Make(self.neural_network, config)()

    def feed_forward (self, inputs) :
        active_input_nodes = [node
                              for input_, node in zip(inputs, self.neural_network.input_nodes())
                              if input_]

        return list(FeedForward(self.neural_network, active_input_nodes))

    def backpropagate (self, inputs, correct, rate) :
        self.feed_forward(inputs)
        return Backpropagate(self.neural_network, correct, rate)()

    def train (self, patterns, loops=10000, rate=0.01) :
        for i in range(loops) :

            error = sum(self.backpropagate(inputs, correct, rate)
                        for inputs, correct in patterns)

            if i % 100 == 0 :
                print('error %-.5f' % error)

    def test (self, patterns) :
        for inputs in patterns :
            yield [int(round(node.charge, 0)) for node in
                   sorted(self.feed_forward(inputs), key=self.neural_network.output_nodes().index)]

def __demo__ () :
    def pat () :
        yield [0, 0, 0, 0], [0, 0, 0, 0]
        yield [1, 1, 0, 1], [1, 1, 1, 1]
        yield [1, 1, 1, 1], [1, 1, 1, 1]
        yield [1, 0, 0, 0], [0, 0, 0, 0]

    pat = [[p, c] for p, c in pat()]

    nn = NN([4, 4, 4])

    for layer in nn.neural_network :
        print(layer)

    nn.train(pat)
    from pprint import pprint
    pprint(list(nn.test([[1, 1, 0, 0]])), width=30)

if __name__ == '__main__' :
    __demo__()

