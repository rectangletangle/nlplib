

from nlplib.core.control.score import Scored
from nlplib.core.process.parse import Parsed
from nlplib.core.model import SessionDependent, NeuralNetwork

def seqs_known_to_neural_network (neural_network) :
    known = set()
    known.update(neural_network.inputs(), neural_network.outputs())
    return known

def __demo__ (ut) :

    from nlplib.data import builtin_db
    from nlplib.core.model import Word, NeuralNetwork

    from nlplib.exterior.train import usable
    from nlplib.exterior.util import plot

    top = 20

    with builtin_db() as session :
        top_words = list(session.access.most_common(Word, top=top)) + [None]

    nn = NeuralNetwork(top_words, len(top_words), top_words)

    with builtin_db() as session :
        patterns = usable(seqs_known_to_neural_network(nn), session.access.all_documents(), gram_size=2)

    plot([nn.train(input_words, output_words, rate=0.1) for input_words, output_words in patterns],
         sample_size=100)

    def ask (string) :
        with builtin_db() as session :
            words = session.access.words(string)

        scored = Scored(nn.predict(words)).sorted()

        print('\n'.join(repr(score) for score in scored))

    return ask

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    ask = __demo__(UnitTest())

