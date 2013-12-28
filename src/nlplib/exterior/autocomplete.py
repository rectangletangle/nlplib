

from nlplib.core.control.neuralnetwork import Prediction, Train, layered
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

        layered.MakeMultilayerPerceptron(neural_network,
                                         layered.static_io(words),
                                         layered.static(len(words)),
                                         layered.static_io(words))()

        return neural_network

def seqs_known_to_neural_network (neural_network) :
    known = set()
    known.update((node.seq for node in neural_network.inputs),
                 (node.seq for node in neural_network.outputs))
    return known

def __demo__ (ut) :
    from pprint import pprint

    from nlplib.core.model import Database, Word, Document, Gram

    from nlplib.exterior.train import usable
    from nlplib.exterior.util import plot_errors

    db = Database()

    top = 40

    with builtin_db() as builtin_session :
        pprint([gram for gram in builtin_session.access.most_common(Gram, 50)], width=20)
        with db as session :
            for word in builtin_session.access.most_common(Word, top=top) :
                session.add(Word(str(word), word.count))
                print(word)

            for total, document in enumerate(builtin_session.access.all_documents()) :
                session.add(Document(str(document)))

    with db as session :
        MakeAutocompleteNeuralNetwork(session, top=top)()

    with db as session :
        nn = session.access.neural_network('autocomplete')
        ut.assert_equal(len(seqs_known_to_neural_network(nn)), top + 1)

        patterns = usable(seqs_known_to_neural_network(nn), list(session.access.all_documents()), gram_size=2)

        plot_errors(Train(session, nn, patterns, 6, rate=0.001), 100)

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


