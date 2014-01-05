

from nlplib.core.control.neuralnetwork import Prediction, Train, structure
from nlplib.core.control.score import Scored
from nlplib.core.process.parse import Parsed
from nlplib.core.model import SessionDependent, NeuralNetwork
from nlplib.data import builtin_db

class Autocomplete (SessionDependent) :
    def __init__ (self, session, neural_network) :
        super().__init__(session)
        self.neural_network = neural_network

    def suggest (self, input_string) :
        return Scored(Prediction(self.session, self.neural_network, tuple(Parsed(input_string))[-3:])).sorted()

    def chose (self, input_, choice) :
        return Train(self.session, self.neural_network, ((input_, choice),), iterations=1, rate=0.01)()

class MakeAutocompleteNeuralNetwork (SessionDependent) :
    def __init__ (self, session, name='autocomplete', top=40) :
        super().__init__(session)
        self.name = name
        self.top = top

    def __call__ (self) :
        neural_network = self.session.add(NeuralNetwork(self.name))

        words = list(self.session.access.most_common(top=self.top)) + [None]

        structure.MakePerceptron(neural_network,
                                 structure.StaticIO(words),
                                 structure.Static(len(words)),
                                 structure.StaticIO(words))()

        return neural_network

def seqs_known_to_neural_network (neural_network) :
    known = set()
    known.update((node.seq for node in neural_network.inputs),
                 (node.seq for node in neural_network.outputs))
    return known

def __demo__ (ut) :
    from pprint import pprint

    from nlplib.core.model import Database, Word, Document, Gram

    from nlplib.core.control.neuralnetwork import numpy_

    from nlplib.exterior.train import usable
    from nlplib.exterior.util import plot

    db = Database()

    top = 2

    with builtin_db() as builtin_session :
        with db as session :
            for word in builtin_session.access.most_common(Word, top=top) :
                session._sqlalchemy_session.merge(word)
                for index in word.indexes :
                    session._sqlalchemy_session.merge(index)

            for total, document in enumerate(builtin_session.access.all_documents()) :
                session._sqlalchemy_session.merge(document)

    with db as session :
        MakeAutocompleteNeuralNetwork(session, top=top)()

    with db as session :
        nn = session.access.neural_network('autocomplete')
        nnn = numpy_.NumpyNeuralNetwork(nn)

        #ut.assert_equal(len(seqs_known_to_neural_network(nn)), top + 1)

        patterns = list(usable(seqs_known_to_neural_network(nn), list(session.access.all_documents())[:2], gram_size=2))
        print([numpy_.NumpyBackpropagate(nnn, in_, out, rate=0.01)() for in_, out in patterns]) # , sample_size=100

    def ask (string) :
        with db as session :
            scored = Scored(Prediction(session,
                                       session.access.nn('autocomplete'),
                                       session.access.words(string))).sorted()

            print('\n'.join(repr(score) for score in scored))

    return ask

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    ask = __demo__(UnitTest())


