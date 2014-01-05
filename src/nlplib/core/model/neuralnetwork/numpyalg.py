''' This module contains basic neural network algorithms implemented using the NumPy library. This allows for
    drastically improved performance over the pure Python implementation. '''


import itertools

import numpy

from nlplib.core.model.neuralnetwork import abstract
from nlplib.core.base import Base
from nlplib.general.iterate import paired
from nlplib.general import composite

__all__ = ['NumpyNeuralNetwork', 'Feedforward', 'Backpropagate']

class NumpyNeuralNetwork (Base) :
    ''' This class provides a way to represent a neural network as a series of NumPy arrays and matrices. '''

    def __init__ (self, neural_network, dtype=float) :
        self.neural_network = neural_network
        self.dtype = dtype

        self.node_indexes = {}
        self.link_indexes = {}

        structure = list(itertools.islice(self._structure(), 1, None))

        self.charges    = structure[::2]
        self.errors     = [numpy.zeros(len(layer), dtype=self.dtype) for layer in self.charges]
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
        ''' This updates the neural network with the contents of the NumPy arrays. '''

        if charges :
            for node, (layer_index, node_index) in self.node_indexes.items() :
                node.charge = self.charges[layer_index][node_index]

        for link, (layer_index, node_index, link_index) in self.link_indexes.items() :
            link.affinity = self.affinities[layer_index-1][node_index, link_index]

    @property
    def inputs (self) :
        ''' Input layer charges. '''

        return self.charges[0]

    @property
    def outputs (self) :
        ''' Output layer charges. '''

        return self.charges[-1]

    def hidden (self, reverse=False) :
        ''' Layers of hidden charges. '''

        return iter(self.charges[1:-1]) if not reverse else reversed(self.charges[1:-1])

    def clear (self) :
        for state in [self.charges, self.errors, self.affinities] :
            for layer_or_links in state :
                layer_or_links.fill(0.0)

class Feedforward (abstract.Feedforward) :
    ''' A fast implementation of the feedforward neural network algorithm, using the NumPy library. '''

    def __call__ (self) :
        self.neural_network.inputs.fill(self.inactive)

        active_indexes = [self.neural_network.node_indexes[node][1] for node in self.active_input_nodes]
        self.neural_network.inputs[active_indexes] = self.active

        dot = numpy.dot
        activation = numpy.vectorize(self.activation)
        as_array = numpy.asarray

        for (input_layer, output_layer), affinities in zip(paired(self.neural_network),
                                                           self.neural_network.affinities) :
            output_layer[:] = activation(as_array(dot(affinities, input_layer)).flatten())

        return output_layer

class Backpropagate (abstract.Backpropagate) :
    ''' A fast implementation of the backpropagation neural network training algorithm, using the NumPy library. '''

    def __call__ (self) :
        output_differences = self._output_errors()
        self._hidden_errors()
        self._update_link_affinities()

        total_error = (0.5 * output_differences ** 2).sum()
        return total_error

    @composite(lambda self : (self.activation_derivative,))
    def _vectorized_activation_derivative (self) :
        return numpy.vectorize(self.activation_derivative)

    def _output_errors (self) :
        activation_derivative = self._vectorized_activation_derivative

        correct = numpy.zeros(len(self.neural_network.outputs))
        indexes = [self.neural_network.node_indexes[node][1] for node in self.correct_output_nodes]

        correct[indexes] = 1.0

        difference = correct - self.neural_network.outputs

        self.neural_network.errors[-1] = activation_derivative(self.neural_network.outputs) * difference

        return difference

    def _hidden_errors (self) :

        dot = numpy.dot
        as_array = numpy.asarray
        activation_derivative = self._vectorized_activation_derivative

        hidden_error_iterator = zip(self.neural_network.hidden(reverse=True),
                                    reversed(self.neural_network.affinities),
                                    reversed(list(paired(self.neural_network.errors))))

        for charges, affinities, (input_errors, output_errors) in hidden_error_iterator :
            difference = as_array(dot(affinities.transpose(), output_errors)).flatten()
            input_errors[:] = activation_derivative(charges) * difference

    def _update_link_affinities (self) :
        link_affinity_update_iterator = zip(itertools.islice(reversed(self.neural_network), 1, None),
                                            reversed(self.neural_network.errors),
                                            reversed(self.neural_network.affinities))

        for input_charges, output_errors, link_affinities in link_affinity_update_iterator :
            link_affinities += self.rate * (input_charges * output_errors[:,numpy.newaxis])

def __test__ (ut) :
    from nlplib.core.control.neuralnetwork.structure import MakePerceptron, StaticIO, Static
    from nlplib.core.model import Database, NeuralNetwork, Word

    db = Database()

    with db as session :
        nn = session.add(NeuralNetwork('foo'))

        for char in 'abcdef' :
            session.add(Word(char))

        MakePerceptron(nn, StaticIO(session.access.words('a b c')), Static(6),
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

                Feedforward(nnn, ins)()
                errors.append(Backpropagate(nnn, ins, outs)())

        ins = [(a,), (b,), (c,), (a, b), (a, c), (b, c)]

        outs = [[ ('f', 0.927115), ('d', 0.082929), ('e', -0.580871) ],
                [ ('d', 0.640232), ('e', 0.179109), ('f', 0.068058)  ],
                [ ('e', 0.691237), ('d', 0.043618), ('f', -0.042884) ],
                [ ('f', 0.854911), ('d', 0.463793), ('e', -0.296235) ],
                [ ('f', 0.85542),  ('d', 0.06778),  ('e', 0.011365)  ],
                [ ('e', 0.669032), ('d', 0.640655), ('f', -0.079214) ]]

        for in_ in ins :

            out = Feedforward(nnn, session.access.input_nodes_for_seqs(nn, in_))()

            strs_and_scores = [(node.object, out[nnn.node_indexes[node][1]]) for node in nn.outputs]

            print(sorted(strs_and_scores, key=lambda both : both[1], reverse=True))

def __profile__ () :
    from nlplib.core.control.neuralnetwork.structure import MakePerceptron, StaticIO, Static
    from nlplib.core.model.neuralnetwork import Feedforward as NLPLibFeedforward, Backpropagate as NLPLibBackpropagate
    from nlplib.core.model import Database, NeuralNetwork, Word
    from nlplib.general import timing
    db = Database()

    size = 100
    loops = 10

    with db as session :
        words = [session.add(Word(str(i))) for i in range(size)]

        nn = session.add(NeuralNetwork('nn'))

        MakePerceptron(nn, StaticIO(words), Static(len(words)), StaticIO(words))()

    with db as session :
        nn  = session.access.neural_network('nn')

        input_ = session.access.input_nodes_for_seqs(nn, session.access.words('0 5 9'))
        output = session.access.output_nodes_for_seqs(nn, session.access.words('0 5 9'))

        @timing
        def numpy_nn () :
            nnn = NumpyNeuralNetwork(nn)
            for _ in range(loops) :
                Feedforward(nnn, input_)()
                Backpropagate(nnn, input_, output)()
            nnn.update()

        @timing
        def nlplib_nn () :
            for _ in range(loops) :
                NLPLibFeedforward(nn, input_)()
                NLPLibBackpropagate(nn, input_, output)()

        numpy_nn()
        nlplib_nn()

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())
    __profile__()


