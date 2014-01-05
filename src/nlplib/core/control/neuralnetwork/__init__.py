

import itertools

from nlplib.core.control.score import Score
from nlplib.core.model import SessionDependent
from nlplib.general import math

__all__ = ['Prediction', 'Train']

class Prediction (SessionDependent) :
    def __init__ (self, session, neural_network, seqs, *args, **kw) :
        super().__init__(session)
        self.neural_network = neural_network
        self.seqs = seqs

        self._args = args
        self._kw = kw

    def __iter__ (self) :
        active_input_nodes = list(self.session.access.nodes_for_seqs(self.neural_network, self.seqs))
        for ouput_node in self.neural_network.feedforward(active_input_nodes, *self._args, **self._kw) :
             yield Score(object=ouput_node.object, score=ouput_node.charge)

class Train (SessionDependent) :
    def __init__ (self, session, neural_network, patterns, iterations=100, rate=0.2, *args, **kw) :
        super().__init__(session)
        self.neural_network = neural_network
        self.patterns = patterns
        self.iterations = iterations
        self.rate = rate

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
                error = self.neural_network.backpropogate(active_input_nodes, correct_output_nodes, rate=rate(i),
                                                          *self._args, **self._kw)
                yield (error, active_input_nodes, correct_output_nodes)

    def __call__ (self) :
        return list(self)

    def _input_and_output_nodes (self) :
        for input_seqs, output_seqs in self.patterns :
            yield (self.session.access.input_nodes_for_seqs(self.neural_network, input_seqs),
                   self.session.access.output_nodes_for_seqs(self.neural_network, output_seqs))

def __test__ (ut) :
    from nlplib.core.control.neuralnetwork.structure import MakePerceptron, StaticIO, Static
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

        MakePerceptron(nn,
                       StaticIO(session.access.words('a b c')),
                       Static(10),
                       StaticIO(session.access.words('d e f')),
                       affinity=affinity)()

    with db as session :
        nn = session.access.neural_network('foo')

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

