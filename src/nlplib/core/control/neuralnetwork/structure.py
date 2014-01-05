

from random import uniform

from nlplib.core.model import Link, Node, IONode
from nlplib.general.iterate import paired, united
from nlplib.core.base import Base
from nlplib.core import exc

__all__ = ['NeuralNetworkConfigurationError', 'NeuralNetworkConfiguration', 'random_affinity', 'MakePerceptron',
           'Static', 'StaticIO']

class NeuralNetworkConfigurationError (exc.NLPLibError) :
    pass

class NeuralNetworkConfiguration (Base) :
    __slots__ = ('_layer_configurations',)

    def __init__ (self, *layer_configurations) :
        if len(layer_configurations) < 2 :
            raise NeuralNetworkConfigurationError('This requires a neural network that has at least two layers.')

        self._layer_configurations = layer_configurations

    def __iter__ (self) :
        return iter(self._layer_configurations)

    def __getitem__ (self, index) :
        return self._layer_configurations[index]

    def hidden (self) :
        return self._layer_configurations[1:-1]

def random_affinity (floor=-1.0, ceiling=1.0) :
    ''' This can be used to initialize the links in the neural network with pseudorandom affinities (weights). '''

    while True :
        yield uniform(floor, ceiling)

class MakePerceptron (Base) :

    def __init__ (self, neural_network, *config, affinity=random_affinity) :
        self.neural_network = neural_network
        self.config = NeuralNetworkConfiguration(*config)
        self.affinity = affinity

    def __call__ (self) :
        for layers in self._link_up(self._layers()) :
            pass

    def __iter__ (self) :
        return united(self._link_up(self._layers()))

    def _layers (self) :
        yield [IONode(self.neural_network, object, is_input=True) for object in self.config[0]]

        for config in self.config[1:-1] :

            if isinstance(config, int) :
                config = Static(config)

            yield [Node(self.neural_network) for _ in config]

        yield [IONode(self.neural_network, object, is_input=False) for object in self.config[-1]]

    def _link_up (self, layers) :
        affinity = self.affinity()

        for input_layer, output_layer in paired(layers) :
            for input_node in input_layer :
                for output_node in output_layer :
                    Link(self.neural_network, input_node, output_node, affinity=next(affinity, 0.0))

            yield (input_layer, output_layer)

class _LayerConfiguration (Base) :
    pass

class Static (_LayerConfiguration) :
    def __init__ (self, size) :
        self.size = size

    def __iter__ (self) :
        for _ in range(self.size) :
            yield None

    def __len__ (self) :
        return self.size

class StaticIO (_LayerConfiguration) :
    def __init__ (self, objects) :
        self.objects = list(objects)

    def __iter__ (self) :
        return iter(self.objects)

    def __len__ (self) :
        return len(self.objects)

class Dynamic (_LayerConfiguration) :
    def __init__ (self, *args, **kw) :
        raise NotImplementedError

class DynamicIO (_LayerConfiguration) :
    def __init__ (self, *args, **kw) :
        raise NotImplementedError

def __test__ (ut) :
    from nlplib.core.model import Database, NeuralNetwork, Word

    db = Database()

    with db as session :
        session.add(NeuralNetwork('foo'))
        for char in 'abcde' :
            session.add(Word(char))

    with db as session :
        MakePerceptron(session.access.neural_network('foo'),
                       StaticIO(session.access.words('a b c')),
                       Static(10),
                       Static(4),
                       Static(5),
                       StaticIO(session.access.words('d e')))()

    with db as session :
        nn = session.access.neural_network('foo')

        ut.assert_equal(sorted(str(node.object) for node in nn.inputs), ['a', 'b', 'c'])

        ut.assert_equal(sorted(str(node.object) for node in nn.outputs), ['d', 'e'])

        for layer, count in zip(nn, [3, 10, 4, 5, 2]) :
            ut.assert_equal(len(layer), count)

        ut.assert_equal(len(list(nn)), 5)

        for layer, count in zip(nn, [0, 3, 10, 4, 5]) :
            for node in layer :
                ut.assert_equal(len(node.inputs), count)

        for layer, count in zip(nn, [10, 4, 5, 2, 0]) :
            for node in layer :
                ut.assert_equal(len(node.outputs), count)

    ut.assert_raises(lambda : MakePerceptron(None, *[]), NeuralNetworkConfigurationError)
    ut.assert_raises(lambda : MakePerceptron(None, *[None]), NeuralNetworkConfigurationError)
    ut.assert_doesnt_raise(lambda : MakePerceptron(None, *[None, None]), NeuralNetworkConfigurationError)

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

