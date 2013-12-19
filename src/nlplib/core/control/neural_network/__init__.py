

import itertools

from nlplib.core.model import SessionDependent
from nlplib.core.score import Score
from nlplib.core import Base, exc
from nlplib.general import math

__all__ = ['FeedForward', 'Backpropagate', 'Prediction', 'Train']

# todo : make math.* functions args

class NeuralNetworkConfigurationError (exc.NLPLibError) :
    pass

class FeedForward (Base) :
    def __init__ (self, neural_network, active_input_nodes, *args, **kw) :
        self.neural_network = neural_network
        self.active_input_nodes = active_input_nodes

    def __call__ (self) :
        for node in self.neural_network.inputs() :
            node.charge = 0.0

        for node in self.active_input_nodes :
            node.charge = 1.0

        for layer in itertools.islice(self.neural_network, 1, None) :
            for node in layer :
                node.charge = math.tanh(self._input_strength(node))

        try :
            return layer
        except UnboundLocalError :
            raise NeuralNetworkConfigurationError('This requires a neural network that has at least two layers.')

    def _input_strength (self, node) :
        return sum(link.input_node.charge * link.affinity for link in node.inputs.values())

class Backpropagate (Base) :
    ''' This is an implementation of the backpropagation algorithm, a supervised neural network training algorithm. '''

    def __init__ (self, neural_network, active_input_nodes, correct_output_nodes, rate=0.2, *args, **kw) :
        self.neural_network = neural_network
        self.active_input_nodes = set(active_input_nodes)
        self.correct_output_nodes = set(correct_output_nodes)
        self.rate = rate

    def __call__ (self) :
        FeedForward(self.neural_network, self.active_input_nodes)()
        return self._propagate_errors()

    def _output_errors (self) :
        for output_node in self.neural_network.outputs() :
            correct_value = 1.0 if output_node in self.correct_output_nodes else 0.0

            difference = correct_value - output_node.charge

            output_node.error = math.dtanh(output_node.charge) * difference

            yield difference

    def _hidden_errors (self) :
        for layer in self.neural_network.hidden(reverse=True) :
            for node in layer :
                node.error = math.dtanh(node.charge) * sum(output_node.error * link.affinity
                                                           for output_node, link in node.outputs.items())

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
    def __init__ (self, session, neural_network, seqs, *args, **kw) :
        super().__init__(session)
        self.neural_network = neural_network
        self.seqs = seqs

    def __iter__ (self) :
        active_input_nodes = list(self.session.access.nodes_for_seqs(self.neural_network, self.seqs))
        for ouput_node in FeedForward(self.neural_network, active_input_nodes)() :

             yield Score(object=ouput_node.seq, score=ouput_node.charge)

class Train (SessionDependent) :
    # todo : make rate take callable

    def __init__ (self, session, neural_network, patterns, iterations=100, rate=0.2, *args, **kw) :
        super().__init__(session)
        self.neural_network = neural_network
        self.patterns = patterns
        self.iterations = iterations
        self.rate = rate

    def __iter__ (self) :
        nodes = list(self._input_and_output_nodes())

        for _ in range(self.iterations) :
            for active_input_nodes, correct_output_nodes in nodes :
                yield Backpropagate(self.neural_network, active_input_nodes, correct_output_nodes, rate=self.rate)()

    def _input_and_output_nodes (self) :
        for input_seqs, output_seqs in self.patterns :

            seqs = tuple(self.session.access.nodes_for_seqs(self.neural_network, input_seqs + output_seqs))

            middle = len(input_seqs)
            yield (seqs[:middle], seqs[middle:])

def __demo__ () :
    from nlplib.core.control.neural_network.layered import MakeMultilayerPerceptron, static_io, static
    from nlplib.core.score import Scored
    from nlplib.core.model import Database, NeuralNetwork, Word

    db = Database()

    with db as session :
        session.add(NeuralNetwork('foo'))
        for char in 'abcdef' :
            session.add(Word(char))

    with db as session :

        config = (static_io(session.access.words('a b c')),)
        config += tuple(static(10) for _ in range(1))
        config += (static_io(session.access.words('d e f')),)

        MakeMultilayerPerceptron(session.access.neural_network('foo'), config)()

    with db as session :
        nn = session.access.neural_network('foo')

        a, b, c, d, e, f = session.access.words('a b c d e f')

        patterns = [((a, b), (f,)),
                    ((a, c), (f,)),
                    ((b,),   (d,)),
                    ((c,),   (e,))]

        for i, error in enumerate(Train(session, nn, patterns, 50, 0.5)) :
            if i % 1 == 0 :
                print(i, ':', error)

        print()

        print('a', sorted(Prediction(session, nn, (a,)), reverse=True))
        print('b', sorted(Prediction(session, nn, (b,)), reverse=True))
        print('c', sorted(Prediction(session, nn, (c,)), reverse=True))
        print()
        print('a b', sorted(Prediction(session, nn, (a, b)), reverse=True))
        print('a c', sorted(Prediction(session, nn, (a, c)), reverse=True))
        print('b c', sorted(Prediction(session, nn, (b, c)), reverse=True))

if __name__ == '__main__' :
    __demo__()

