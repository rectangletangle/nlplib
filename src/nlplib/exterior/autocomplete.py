

from nlplib.core.control.score import Scored
from nlplib.core.process.parse import Parsed
from nlplib.core.model import SessionDependent, NeuralNetwork

from nlplib.general import timing
@timing
def __demo__ (ut) :

    from nlplib.data import builtin_db
    from nlplib.core.model import Word, NeuralNetwork

    from nlplib.exterior.train import usable
    from nlplib.exterior.util import plot

    top = 10

    with builtin_db() as session :
        top_words = list(session.access.most_common(Word, top=top)) + [None]

    nn = NeuralNetwork(top_words, int(len(top_words)*2), top_words)

    with builtin_db() as session :
        patterns = usable(nn, list(session.access.all_documents())[:], gram_size=2)

    errors = [nn.train(input_words, output_words, rate=0.1) for input_words, output_words in patterns]

    from nlplib.general.math import avg

    print(avg(errors[:]))

    plot(errors,
         sample_size=100)

    def ask (string) :
        with builtin_db() as session :
            words = session.access.words(string)

        scored = Scored(nn.predict(words)).sorted()

        print('\n'.join(repr(score) for score in list(scored)[:12]))

    return ask

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    ask = __demo__(UnitTest())

