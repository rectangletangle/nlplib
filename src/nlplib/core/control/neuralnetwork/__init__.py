

import itertools

from nlplib.core.control.score import Score
from nlplib.core.model import SessionDependent
from nlplib.core.base import Base
from nlplib.general import math

__all__ = ['FeedForward', 'Backpropagate', 'Prediction', 'Train']

class FeedForward (Base) :
    def __init__ (self, neural_network, active_input_nodes, active=1.0, inactive=0.0, activation=math.tanh) :
        self.neural_network = neural_network
        self.active_input_nodes = active_input_nodes

        self.active   = active
        self.inactive = inactive

        self.activation = activation

    def __call__ (self) :

        self._set_charges(self.neural_network.inputs, self.inactive)

        self._set_charges(self.active_input_nodes, self.active)

        for layer in itertools.islice(self.neural_network, 1, None) :
            for node in layer :
                node.charge = self.activation(self._input_strength(node))

        return layer

    def _set_charges (self, nodes, charge) :
        if callable(charge) :
            for node in nodes :
                node.charge = charge(node)
        else :
            for node in nodes :
                node.charge = charge

    def _input_strength (self, node) :
        return sum(input_node.charge * link.affinity for input_node, link in node.inputs.items())

class Backpropagate (Base) :
    ''' This is an implementation of the backpropagation algorithm, a supervised neural network training algorithm.

        Note : Because this algorithm depends on dictionary iteration order in several places, it should be considered
        nondeterministic. '''

    def __init__ (self, neural_network, active_input_nodes, correct_output_nodes, rate=0.2,
                  activation_derivative=math.dtanh) :

        self.neural_network = neural_network
        self.active_input_nodes = set(active_input_nodes)
        self.correct_output_nodes = set(correct_output_nodes)
        self.rate = rate
        self.activation_derivative = activation_derivative

    def __call__ (self) :
        FeedForward(self.neural_network, self.active_input_nodes)()
        return self._propagate_errors()

    def _output_errors (self) :
        for output_node in self.neural_network.outputs :
            correct_value = 1.0 if output_node in self.correct_output_nodes else 0.0

            difference = correct_value - output_node.charge

            output_node.error = self.activation_derivative(output_node.charge) * difference

            yield difference

    def _hidden_errors (self) :
        for layer in self.neural_network.hidden(reverse=True) :
            for node in layer :
                node.error = self.activation_derivative(node.charge) * sum(output_node.error * link.affinity
                                                                           for output_node, link
                                                                           in node.outputs.items())

    def _update_link_affinities (self) :
        for layer in itertools.islice(reversed(self.neural_network), 1, None) :
            for node in layer :
                for link in node.outputs.values() :
                    link.affinity += self.rate * link.output_node.error * node.charge

    def _propagate_errors (self) :
        output_differences = list(self._output_errors())

        self._hidden_errors()

        self._update_link_affinities()

        total_error = sum(0.5 * difference ** 2 for difference in output_differences)

        return total_error

class Prediction (SessionDependent) :
    def __init__ (self, session, neural_network, seqs, algorithm=FeedForward, *args, **kw) :
        super().__init__(session)
        self.neural_network = neural_network
        self.seqs = seqs
        self.algorithm = algorithm

        self._args = args
        self._kw = kw

    def __iter__ (self) :
        active_input_nodes = list(self.session.access.nodes_for_seqs(self.neural_network, self.seqs))
        for ouput_node in self.algorithm(self.neural_network, active_input_nodes, *self._args, **self._kw)() :
             yield Score(object=ouput_node.seq, score=ouput_node.charge)

class Train (SessionDependent) :
    def __init__ (self, session, neural_network, patterns, iterations=100, rate=0.2, algorithm=Backpropagate,
                  *args, **kw) :
        super().__init__(session)
        self.neural_network = neural_network
        self.patterns = patterns
        self.iterations = iterations
        self.rate = rate
        self.algorithm = algorithm

        self._args = args
        self._kw = kw

    def __iter__ (self) :

        if callable(self.rate) :
            rate = self.rate
        else :
            rate = lambda i : self.rate

        nodes = list(self._input_and_output_nodes())

        for i in range(self.iterations) :
            for active_input_nodes, correct_output_nodes in nodes :
                error = self.algorithm(self.neural_network, active_input_nodes, correct_output_nodes, rate=rate(i),
                                       *self._args, **self._kw)()

                yield (error, active_input_nodes, correct_output_nodes)

    def __call__ (self) :
        return list(self)

    def _input_and_output_nodes (self) :
        for input_seqs, output_seqs in self.patterns :
            yield (self.session.access.input_nodes_for_seqs(self.neural_network, input_seqs),
                   self.session.access.output_nodes_for_seqs(self.neural_network, output_seqs))

def __test__ (ut) :
    from collections import OrderedDict

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
                                 Static(10),
                                 StaticIO(session.access.words('d e f')),
                                 affinity=affinity)()

    with db as session :
        nn = session.access.neural_network('foo')

        def deterministic_dict (dict_) :
            # Because dictionary iteration order is undefined behavior, and in some Python implementations not even
            # deterministic, we introduce deterministic iteration order to the algorithm using the ordered dictionary
            # type.

            return OrderedDict(sorted(dict_.items(), key=lambda item : item[1].affinity))

        for node in session.access.nodes(nn) :
            node.inputs  = deterministic_dict(node.inputs)
            node.outputs = deterministic_dict(node.outputs)

        a, b, c, d, e, f = session.access.words('a b c d e f')

        training_patterns = [( (a, b), (f,) ),
                             ( (a, c), (f,) ),
                             ( (b,),   (d,) ),
                             ( (c,),   (e,) )]

        avg_error = math.avg(round(error, 6) for error, *nodes in Train(session, nn, training_patterns, 20, 0.2))

        ut.assert_equal(avg_error, 0.4993055999999999)

        # We never directly trained it for the input combination (b, c); however, it throws out a reasonable guess by
        # giving e and d a similar score, and giving f a low score.
        ins = [(a,), (b,), (c,), (a, b), (a, c), (b, c)]

        outs = [[ ('f', 0.927115), ('d', 0.082929), ('e', -0.580871) ],
                [ ('d', 0.640232), ('e', 0.179109), ('f', 0.068058)  ],
                [ ('e', 0.691237), ('d', 0.043618), ('f', -0.042884) ],
                [ ('f', 0.854911), ('d', 0.463793), ('e', -0.296235) ],
                [ ('f', 0.85542),  ('d', 0.06778),  ('e', 0.011365)  ],
                [ ('e', 0.669032), ('d', 0.640655), ('f', -0.079214) ]]

        for in_, out in zip(ins, outs) :
            strs_and_scores = [(str(word), round(score, 6)) for word, score in Prediction(session, nn, in_)]

            ut.assert_equal(sorted(strs_and_scores, key=lambda both : both[1], reverse=True),
                            out)

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

