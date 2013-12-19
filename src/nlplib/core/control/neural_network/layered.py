

from random import uniform

from nlplib.core.model import Link, Node, IONode
from nlplib.general.iter import paired
from nlplib.core import Base

# todo : implement __all__ list
# todo : should be callable not gen type

def random_affinity (floor=-1.0, ceiling=1.0) :
    ''' This can be used to initialize the links in the neural network with pseudorandom affinities (weights). '''

    while True :
        yield uniform(floor, ceiling)

class MakeMultilayerPerceptron (Base) :
    # todo : make it so only a repr of this class needs to be stored in the db, instead of all these node instances.
    # faster to load from db, less data stored in db, use "dynamic" in name

    def __init__ (self, neural_network, config, affinity=random_affinity) :
        self.neural_network = neural_network
        self.config = config
        self.affinity = affinity

    def __call__ (self) :
        self.link_up(self.layers())

    def layers (self) :

        yield [IONode(self.neural_network, seq, is_input=True) for seq in self.config[0]]

        for config in self.config[1:-1] :

            if isinstance(config, int) :
                config = static(config)

            yield [Node(self.neural_network) for _ in config]

        yield [IONode(self.neural_network, seq, is_input=False) for seq in self.config[-1]]

    def link_up (self, layers) :
        affinity = self.affinity()

        for input_layer, output_layer in paired(layers) :
            for input_node in input_layer :
                for output_node in output_layer :
                    Link(self.neural_network, input_node, output_node, affinity=next(affinity, 0.0))

def static (size) :
    for _ in range(size) :
        yield None

def static_io (seqs) :
    for seq in seqs :
        yield seq

def dynamic () :
    raise NotImplementedError

def __test__ (ut) :
    from nlplib.core.model import Database, NeuralNetwork, Word

    db = Database()

    with db as session :
        session.add(NeuralNetwork('foo'))
        for char in 'abcde' :
            session.add(Word(char))

    with db as session :

        config = [static_io(session.access.words('a b c')),
                  static(10),
                  static(4),
                  static(5),
                  static_io(session.access.words('d e'))]

        MakeMultilayerPerceptron(session.access.neural_network('foo'), config)()

    with db as session :
        nn = session.access.neural_network('foo')

        ut.assert_equal(sorted(str(node.seq) for node in nn.inputs()), ['a', 'b', 'c'])

        ut.assert_equal(sorted(str(node.seq) for node in nn.outputs()), ['d', 'e'])

        for layer, count in zip(nn, [3, 10, 4, 5, 2]) :
            ut.assert_equal(len(layer), count)

        ut.assert_equal(len(list(nn)), 5)

        for layer, count in zip(nn, [0, 3, 10, 4, 5]) :
            for node in layer :
                ut.assert_equal(len(node.inputs), count)

        for layer, count in zip(nn, [10, 4, 5, 2, 0]) :
            for node in layer :
                ut.assert_equal(len(node.outputs), count)

if __name__ == '__main__' :
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())

