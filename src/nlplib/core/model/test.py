''' This tests the models. '''


import itertools

from nlplib.core.process.index import Indexed
from nlplib.core.model import Database, Document, Word, NeuralNetwork, Layer, NeuralNetworkIO

def _test_document (ut) :
    # Tests the addition and removal of documents and associated objects from the database.

    db = Database()

    with db as session :
        indexed = Indexed(session)
        indexed.add(session.add(Document('a b b a c d')), max_gram_length=1)
        indexed.add(session.add(Document('a c d e')), max_gram_length=2)

        session.add(Document('a c')) # This document isn't indexed, so no associated objects.

    def test_counts (session, seq, index) :
        ut.assert_equal(len(list(session.access.all_seqs())), seq)
        ut.assert_equal(len(list(session.access.all_indexes())), index)

    def longest (session) :
        return max(session.access.all_documents(), key=lambda document : len(str(document)))

    with db as session :
        documents = list(session.access.all_documents())
        ut.assert_equal(len(documents), 3)

        ut.assert_true(session.access.word('a') in documents[0])
        ut.assert_true(session.access.word('e') not in documents[0])

        ut.assert_true(session.access.word('a') in documents[1])
        ut.assert_true(session.access.word('e') in documents[1])
        ut.assert_true(session.access.word('b') not in documents[1])

        ut.assert_true(session.access.word('a') not in documents[2])
        ut.assert_true(session.access.word('c') not in documents[2])
        ut.assert_true(session.access.word('e') not in documents[2])

        test_counts(session, 8, 13)
        session.remove(longest(session))
        test_counts(session, 7, 7)
        session.remove(longest(session))
        test_counts(session, 0, 0)
        session.remove(longest(session))
        test_counts(session, 0, 0)

        ut.assert_equal(len(list(session.access.all_documents())), 0)

    with db as session :
        test_counts(session, 0, 0)

def _test_neural_network_io (ut) :
    # Neural networks IO objects should be able to handle sequences, <None>, or pickle-able Python objects as their
    # object property.

    db = Database()

    with db as session :
        session.add(Word('foo'))

    with db as session :
        structure = session.add(NeuralNetwork(name='bar'))._structure

        layer = Layer(structure)

        append = layer.io.append
        append(NeuralNetworkIO(structure, None))
        append(NeuralNetworkIO(structure, ''))
        append(NeuralNetworkIO(structure, 'baz'))
        append(NeuralNetworkIO(structure, session.access.word('foo')))
        append(NeuralNetworkIO(structure, [1, 2, 'hello']))
        append(NeuralNetworkIO(structure, {'a' : 0, 'b' : 1, 2 : 'c'})) # The <2> key is converted into a string.

    with db as session :
        nn = session.access.neural_network('bar')
        ut.assert_equal(list(nn._structure.inputs().objects()),
                        [None, '', 'baz', session.access.word('foo'), [1, 2, 'hello'], {'a' : 0, 'b' : 1, '2' : 'c'}])


def _test_neural_network_methods (ut) :

    training_patterns = [('ab', 'f'),
                         ('ac', 'f'),
                         ('b',  'd'),
                         ('c',  'e')]

    def weights () :
        # Convoluted, yet deterministic weights.
        for i in itertools.count() :
            yield ((23 / 7) * 3.01 * i / 283) * (-1 if i % 2 == 0 else 1)

    nn = NeuralNetwork('abc', 3, 'def', weights=weights)

    total_error = round(sum(nn.train(inputs, outputs) for _ in range(20) for inputs, outputs in training_patterns), 5)

    ut.assert_equal(total_error, 21.62668)

    correct_outputs = [('f', 'd', 'e'),
                       ('d', 'f', 'e'),
                       ('e', 'f', 'd'),
                       ('f', 'd', 'e'),
                       ('f', 'e', 'd'),
                       ('e', 'f', 'd')]

    for inputs, correct in zip(['a', 'b', 'c', 'ab', 'ac', 'bc'], correct_outputs) :
        ut.assert_equal([object for object, score in sorted(nn.predict(inputs), reverse=True)], correct)

    ut.assert_true('a' in nn)
    ut.assert_true('d' in nn)
    ut.assert_true('z' not in nn)
    ut.assert_true(173 not in nn)

    ut.assert_equal(list(nn.inputs()), list('abc'))
    ut.assert_equal(list(nn.outputs()), list('def'))
    ut.assert_equal(list(nn), list('abcdef'))
    ut.assert_equal([(object, round(score, 5)) for object, score in nn.scores()],
                    [('d', -0.04185), ('e', 0.12802), ('f', 0.11855)])

def __test__ (ut) :
    _test_document(ut)
    _test_neural_network_io(ut)
    _test_neural_network_methods(ut)

    from nlplib.core.model import Database, Layer

    from sqlalchemy import inspect
    from sqlalchemy import BLOB


    from nlplib.core.model.sqlalchemy_.map import default_mapped
    from nlplib.general import timing

    size = 10

    db = Database()

    @timing
    @db
    def bar (session) :
        #session.add(NeuralNetwork(list('abc'), list(range(456)), list('def'), name='foo'))
        session.add(NeuralNetwork([{0 : 'a', 'b' : 1}, {}, 324, b'\xc3\x9c'.decode(), 'a', 'b'],
                                  size,
                                  range(size),
                                  name='foo'))


        nn = session.access.nn('foo')
        print('d' in nn)
        print('sdf' in nn)
        print(list(nn.inputs()))
        print(list(nn.outputs()))
        print(list(nn))
        print(list(nn.scores()))

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

