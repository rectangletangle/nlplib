

import itertools

from nlplib.core.control.score import Score
from nlplib.core.model import SessionDependent
from nlplib.core.base import Base
from nlplib.general import math

__all__ = ['FeedForward', 'Backpropagate', 'Prediction', 'Train']

# todo : make math.* functions args

class FeedForward (Base) :
    def __init__ (self, neural_network, active_input_nodes, *args, **kw) :
        self.neural_network = neural_network
        self.active_input_nodes = active_input_nodes

    def __call__ (self) :
        for node in self.neural_network.inputs :
            node.charge = 0.0

        for node in self.active_input_nodes :
            node.charge = 1.0

        for layer in itertools.islice(self.neural_network, 1, None) :
            for node in layer :
                node.charge = math.tanh(self._input_strength(node))

        return layer

    def _input_strength (self, node) :
        return sum(input_node.charge * link.affinity for input_node, link in node.inputs.items())

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
        for output_node in self.neural_network.outputs :
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

    def __call__ (self) :
        return list(self)

    def _input_and_output_nodes (self) :
        for input_seqs, output_seqs in self.patterns :

            seqs = tuple(self.session.access.nodes_for_seqs(self.neural_network, input_seqs + output_seqs))

            middle = len(input_seqs)
            yield (seqs[:middle], seqs[middle:])

def __test__ (ut) :
    from nlplib.core.control.neuralnetwork.layered import MakeMultilayerPerceptron, static_io, static, random_affinity
    from nlplib.core.control.score import Scored
    from nlplib.core.model import Database, NeuralNetwork, Word

    # todo : not deterministic, probably set order doing it

    db = Database()

    with db as session :
        session.add(NeuralNetwork('foo'))
        for char in 'abcdef' :
            session.add(Word(char))

    with db as session :

        config = (static_io(session.access.words('a b c')),)
        config += tuple(static(10) for _ in range(1))
        config += (static_io(session.access.words('d e f')),)

        def foo (*args) :
            while True :
                for i in range(10) :
                    u = i / 10
                    yield u if i % 2 == 0 else u * -1

        MakeMultilayerPerceptron(session.access.neural_network('foo'), config, affinity=foo)()

    with db as session :
        nn = session.access.neural_network('foo')

        a, b, c, d, e, f = session.access.words('a b c d e f')

        patterns = [((a, b), (f,)),
                    ((a, c), (f,)),
                    ((b,),   (d,)),
                    ((c,),   (e,))]

        correct_errors = [0.992051, 0.991895, 0.929592, 2.853537, 0.989829, 0.988991, 0.911346, 2.801319, 0.985935,
                          0.983401, 0.881609, 2.704446, 0.977543, 0.969563, 0.825323, 2.485274, 0.950553, 0.913019,
                          0.682092, 1.80601, 0.767657, 0.48561, 0.296976, 0.382413, 0.463988, 0.493707, 0.26755,
                          0.151701, 0.436848, 0.361061, 0.2944, 0.145551, 0.411768, 0.244382, 0.305417, 0.159377,
                          0.388434, 0.174774, 0.27522, 0.161916, 0.371833, 0.137788, 0.234883, 0.157282, 0.359215,
                          0.114118, 0.201198, 0.148665, 0.346272, 0.097993, 0.176928, 0.137895, 0.330956, 0.087816,
                          0.161396, 0.126661, 0.312308, 0.082303, 0.152689, 0.116299, 0.290045, 0.080179, 0.14864,
                          0.107533, 0.264584, 0.080117, 0.147227, 0.100513, 0.237145, 0.080721, 0.146654, 0.095013,
                          0.209627, 0.080719, 0.145447, 0.090641, 0.18408, 0.079332, 0.142691, 0.086964]

        #print([round(error, 6) for error in Train(session, nn, patterns, 20, 0.2)] == correct_errors)

        ins_and_outs = [( (a,),   [('f', 0.893482), ('d', -0.085247), ('e', -0.22805)] ),
                        ( (b,),   [('d', 0.628165), ('e', 0.106708), ('f', -0.063184)] ),
                        ( (c,),   [('e', 0.708942), ('d', 0.090739), ('f', 0.046273)]  ),
                        ( (a, b), [('f', 0.875242), ('d', 0.358109), ('e', -0.304012)] ),
                        ( (a, c), [('f', 0.837407), ('e', 0.09891), ('d', -0.137624)]  ),
                        ( (b, c), [('e', 0.698783), ('d', 0.673296), ('f', 0.10256)]   )]

        for in_, out in ins_and_outs :
            strs_and_scores  = [(str(word), round(score, 6)) for word, score in Prediction(session, nn, in_)]
            sorted_by_scores = sorted(strs_and_scores, key=lambda both : both[1] * -1)
            #print(sorted_by_scores)
            #ut.assert_equal(sorted_by_scores, out)

def __profile__ () :
    from nlplib.core.control.neuralnetwork.layered import MakeMultilayerPerceptron, static_io, static
    from nlplib.core.control.score import Scored
    from nlplib.core.model import Database, NeuralNetwork, Word

    db = Database()

    with db as session :
        session.add(NeuralNetwork('foo'))
        for char in range(10) :
            session.add(Word(str(char)))

    with db as session :

        config = (static_io(session.access.words(' '.join(str(i) for i in range(5)))),)
        config += tuple(static(10) for _ in range(1))
        config += tuple(static(4) for _ in range(1))
        config += (static_io(session.access.words(' '.join(str(i) for i in range(5, 10)))),)

        MakeMultilayerPerceptron(session.access.neural_network('foo'), config)()

    from nlplib.general.time import timing
    @timing
    @db.session
    def t (session) :
        pass


##        nn = session.access.neural_network('foo')
##
##        a, b, c, d, e, f = session.access.words('a b c d e f')
##
##        patterns = [((a, b), (f,)),
##                    ((a, c), (f,)),
##                    ((b,),   (d,)),
##                    ((c,),   (e,))]
##
##        for error in Train(session, nn, patterns, 1, 0.5) :
##            pass

    t()

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())
    #__profile__()

