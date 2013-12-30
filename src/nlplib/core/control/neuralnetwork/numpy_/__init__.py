

import itertools

import numpy

from nlplib.core.base import Base

__all__ = ['NumpyNeuralNetwork']

class NumpyNeuralNetwork (Base) :
    ''' This class provides a way to represent a neural network as a series of numpy arrays. Allowing the use of
        numpy's blazing fast number crunching abilities. '''

    def __init__ (self, neural_network, dtype=float) :
        self.neural_network = neural_network
        self.dtype = dtype

        self.node_indexes = {}
        self.link_indexes = {}

        self.layers      = list(self._make_layer_arrays())
        self.connections = list(self._make_connection_matrixes())

    def _make_layer_arrays (self) :
        for layer_index, layer in enumerate(self.neural_network) :
            charges = []
            charges_append = charges.append

            for node_index, node in enumerate(layer) :
                self.node_indexes[node] = (layer_index, node_index)
                charges_append(node.charge)

            yield numpy.array(charges, dtype=self.dtype)

    def _make_connection_matrixes (self) :
        for intralayer_index, layer in enumerate(itertools.islice(self.neural_network, 1, None)) :
            connections = []
            connections_append = connections.append

            for node_index, node in enumerate(layer) :
                affinities = []
                affinities_append = affinities.append

                for link_index, link in enumerate(node.inputs.values()) :
                    self.link_indexes[link] = (intralayer_index, node_index, link_index)
                    affinities_append(link.affinity)

                connections_append(affinities)

            yield numpy.matrix(connections, dtype=self.dtype)

    def update (self, charges=True) :
        ''' This updates the neural network with the contents of the numpy arrays. '''

        if charges :
            for node, (layer_index, node_index) in self.node_indexes.items() :
                node.charge = self.layers[layer_index][node_index]

        for link, (intralayer_index, node_index, link_index) in self.link_indexes.items() :
            link.affinity = self.connections[intralayer_index][node_index, link_index]

def __test__ (ut) :
    from nlplib.core.control.neuralnetwork.layered import MakeMultilayerPerceptron, StaticIO, Static
    from nlplib.core.control.score import Scored
    from nlplib.core.model import Database, NeuralNetwork, Word

    db = Database()

    with db as session :
        session.add(NeuralNetwork('foo'))
        for char in range(8) :
            session.add(Word(str(char)))

    with db as session :

        config = (StaticIO(session.access.words(' '.join(str(i) for i in range(5)))),
                  Static(10),
                  Static(4),
                  Static(17),
                  StaticIO(session.access.words(' '.join(str(i) for i in range(5, 8)))))

        MakeMultilayerPerceptron(session.access.neural_network('foo'), *config)()

    with db as session :
        nn = session.access.neural_network('foo')

        nnn = NumpyNeuralNetwork(nn)

        for layer, size in zip(nnn.layers, [5, 10, 4, 17, 3]) :
            ut.assert_equal(list(layer), [0.0] * size)

        for layer in nnn.layers :
            layer += 3 * len(layer)

        nnn.update()

        #for layer, size in zip(nnn.layers, [5, 10, 4, 17, 3]) :
        #    ut.assert_equal(list(layer), [0.0] * size)

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())
