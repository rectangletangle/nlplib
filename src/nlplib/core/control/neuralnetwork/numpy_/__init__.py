

import itertools

import numpy

from nlplib.core.base import Base
from nlplib.general.iterate import paired
from nlplib.general import math, composite

__all__ = ['NumpyNeuralNetwork']

class NumpyNeuralNetwork (Base) :
    ''' This class provides a way to represent a neural network as a series of numpy arrays and matrices, allowing for
        the use of numpy's blazing fast number crunching. '''

    def __init__ (self, neural_network, dtype=float) :
        self.neural_network = neural_network
        self.dtype = dtype

        self.node_indexes = {}
        self.link_indexes = {}

        structure = list(itertools.islice(self._structure(), 1, None))

        self.charges = structure[::2]
        self.errors  = [numpy.zeros(len(layer)) for layer in self.charges]

        self.affinities = structure[1::2]

    def _structure (self) :
        for layer_index, layer in enumerate(self.neural_network) :
            charges = []
            charges_append = charges.append

            links = []
            links_append = links.append

            for node_index, node in enumerate(layer) :
                self.node_indexes[node] = (layer_index, node_index)
                charges_append(node.charge)

                affinities = []
                affinities_append = affinities.append

                for link_index, link in enumerate(node.inputs.values()) :
                    self.link_indexes[link] = (layer_index, node_index, link_index)
                    affinities_append(link.affinity)

                links_append(affinities)

            yield numpy.matrix(links, dtype=self.dtype)
            yield numpy.array(charges, dtype=self.dtype)

    def __iter__ (self) :
        return iter(self.charges)

    def __reversed__ (self) :
        return reversed(self.charges)

    def update (self, charges=True) :
        ''' This updates the neural network with the contents of the numpy arrays. '''

        if charges :
            for node, (layer_index, node_index) in self.node_indexes.items() :
                node.charge = self.charges[layer_index][node_index]

        for link, (layer_index, node_index, link_index) in self.link_indexes.items() :
            link.affinity = self.affinities[layer_index-1][node_index, link_index]

    @property
    def inputs (self) :
        return self.charges[0]

    @property
    def outputs (self) :
        return self.charges[-1]

    def hidden (self, reverse=False) :
        return iter(self.charges[1:-1]) if not reverse else reversed(self.charges[1:-1])

class NumpyFeedForward (Base) :
    def __init__ (self, numpy_neural_network, active_input_nodes, active=1.0, inactive=0.0, activation=math.tanh) :
        self.numpy_neural_network = numpy_neural_network
        self.active_input_nodes = active_input_nodes

        self.active   = active
        self.inactive = inactive

        self.activation = activation

    def __call__ (self) :

        self.numpy_neural_network.inputs.fill(self.inactive)

        active_indexes = [self.numpy_neural_network.node_indexes[node][1] for node in self.active_input_nodes]
        self.numpy_neural_network.inputs[active_indexes] = self.active

        dot = numpy.dot
        activation = numpy.vectorize(self.activation)
        as_array = numpy.asarray

        for (input_layer, output_layer), affinities in zip(paired(self.numpy_neural_network),
                                                           self.numpy_neural_network.affinities) :
            output_layer[:] = activation(as_array(dot(affinities, input_layer)).flatten())
        return output_layer

class NumpyBackpropagate (Base) :

    def __init__ (self, numpy_neural_network, active_input_nodes, correct_output_nodes, rate=0.2,
                  activation_derivative=math.dtanh) :

        self.numpy_neural_network = numpy_neural_network

        self.active_input_nodes = set(active_input_nodes)
        self.correct_output_nodes = set(correct_output_nodes)
        self.rate = rate
        self.activation_derivative = activation_derivative

    def __call__ (self) :
        NumpyFeedForward(self.numpy_neural_network, self.active_input_nodes)()
        return self._propagate_errors()

    @composite(lambda self : (self.activation_derivative,))
    def _vectorized_activation_derivative (self) :
        return numpy.vectorize(self.activation_derivative)

    def _output_errors (self) :
        activation_derivative = self._vectorized_activation_derivative

        correct = numpy.zeros(len(self.numpy_neural_network.outputs))
        indexes = [self.numpy_neural_network.node_indexes[node][1] for node in self.correct_output_nodes]

        correct[indexes] = 1.0

        difference = correct - self.numpy_neural_network.outputs

        self.numpy_neural_network.errors[-1] = activation_derivative(self.numpy_neural_network.outputs) * difference

        return difference

    def _hidden_errors (self) :

        dot = numpy.dot
        as_array = numpy.asarray
        activation_derivative = self._vectorized_activation_derivative

        hidden_error_iterator = zip(self.numpy_neural_network.hidden(reverse=True),
                                    reversed(list(paired(self.numpy_neural_network.errors))),
                                    reversed(self.numpy_neural_network.affinities))

        for charges, (input_errors, output_errors), affinities in hidden_error_iterator :
            difference = as_array(dot(affinities.transpose(), output_errors)).flatten()
            input_errors[:] = activation_derivative(charges) * difference

    def _update_link_affinities (self) :
        link_affinity_update_iterator = zip(itertools.islice(reversed(self.numpy_neural_network), 1, None),
                                            reversed(self.numpy_neural_network.errors),
                                            reversed(self.numpy_neural_network.affinities))

        for input_charges, output_errors, link_affinities in link_affinity_update_iterator :
            link_affinities += self.rate * (input_charges * output_errors[:,numpy.newaxis])

    def _propagate_errors (self) :
        output_differences = self._output_errors()
        self._hidden_errors()
        self._update_link_affinities()

        total_error = (0.5 * output_differences ** 2).sum()
        return total_error

def __test__ (ut) :
    from nlplib.core.control.neuralnetwork.layered import MakeMultilayerPerceptron, StaticIO, Static
    from nlplib.core.control.score import Scored
    from nlplib.core.model import Database, NeuralNetwork, Word

    db = Database()

    s = 3

    with db as session :
        session.add(NeuralNetwork('foo'))
        for char in range(s) :
            session.add(Word(str(char)))

    with db as session :

        config = (StaticIO(session.access.words(' '.join(str(i) for i in range(s)))),
                  Static(s),
                  StaticIO(session.access.words(' '.join(str(i) for i in range(s)))))

        nn = session.access.neural_network('foo')

        MakeMultilayerPerceptron(nn, *config)()

    with db as session :
        nn = session.access.neural_network('foo')

        nnn = NumpyNeuralNetwork(nn)

        input_ = (nn.inputs[0],)
        from nlplib.exterior.util import plot

        plot([NumpyBackpropagate(nnn, input_, (nn.outputs[0], nn.outputs[-1]))() for _ in range(10)])

def __test__ (ut) :
    from nlplib.core.control.neuralnetwork.layered import MakeMultilayerPerceptron, StaticIO, Static
    from nlplib.core.model import Database, NeuralNetwork, Word

    db = Database()

    with db as session :
        nn = session.add(NeuralNetwork('foo'))

        for char in 'abcdef' :
            session.add(Word(char))

        def affinity () :
            # This is used in place of random affinities, in order to maintain determinism.

            while True :
                for i in range(10) :
                    u = i / 10
                    yield u if i % 2 == 0 else u * -1

        MakeMultilayerPerceptron(nn,
                                 StaticIO(session.access.words('a b c')),
                                 Static(6),
                                 StaticIO(session.access.words('d e f')))()

    with db as session :
        nn = session.access.neural_network('foo')

        nnn = NumpyNeuralNetwork(nn)

        a, b, c, d, e, f = session.access.words('a b c d e f')

        training_patterns = [( (a, b), (f,) ),
                             ( (a, c), (f,) ),
                             ( (b,),   (d,) ),
                             ( (c,),   (e,) )]


        errors = []
        for _ in range(100) :
            for in_, out in training_patterns :

                ins  = session.access.input_nodes_for_seqs(nn, in_)
                outs = session.access.output_nodes_for_seqs(nn, out)

                errors.append(NumpyBackpropagate(nnn, ins, outs)())

        ins = [(a,), (b,), (c,), (a, b), (a, c), (b, c)]

        outs = [[ ('f', 0.927115), ('d', 0.082929), ('e', -0.580871) ],
                [ ('d', 0.640232), ('e', 0.179109), ('f', 0.068058)  ],
                [ ('e', 0.691237), ('d', 0.043618), ('f', -0.042884) ],
                [ ('f', 0.854911), ('d', 0.463793), ('e', -0.296235) ],
                [ ('f', 0.85542),  ('d', 0.06778),  ('e', 0.011365)  ],
                [ ('e', 0.669032), ('d', 0.640655), ('f', -0.079214) ]]

        for in_ in ins :

            out = NumpyFeedForward(nnn, session.access.input_nodes_for_seqs(nn, in_))()

            strs_and_scores = [(node.seq, out[nnn.node_indexes[node][1]]) for node in nn.outputs]

            #print(sorted(strs_and_scores, key=lambda both : both[1], reverse=True))

def __profile__ () :
    from nlplib.core.control.neuralnetwork.layered import MakeMultilayerPerceptron, StaticIO, Static
    from nlplib.core.control.neuralnetwork import Backpropagate
    from nlplib.core.model import Database, NeuralNetwork, Word
    from nlplib.general import timing
    db = Database()

    size = 40
    loops = 3

    with db as session :
        words = [session.add(Word(str(i))) for i in range(size)]

        nn = session.add(NeuralNetwork('nn'))

        MakeMultilayerPerceptron(nn,
                                 StaticIO(words),
                                 Static(len(words)),
                                 StaticIO(words))()

    with db as session :
        nn  = session.access.neural_network('nn')

        input_ = session.access.input_nodes_for_seqs(nn, session.access.words('0 5 9'))
        output = session.access.output_nodes_for_seqs(nn, session.access.words('0 5 9'))

        @timing
        def numpy_nn () :
            nnn = NumpyNeuralNetwork(nn)
            for _ in range(loops) :
                NumpyBackpropagate(nnn, input_, output)()
            nnn.update()

        @timing
        def nlplib_nn () :
            for _ in range(loops) :
                Backpropagate(nn, input_, output)()

        numpy_nn()
        nlplib_nn()

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())
    __profile__()


