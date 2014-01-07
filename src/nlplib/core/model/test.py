''' This tests the models. '''

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

        NeuralNetworkIO(layer, None)
        NeuralNetworkIO(layer, '')
        NeuralNetworkIO(layer, 'baz')
        NeuralNetworkIO(layer, session.access.word('foo'))
        NeuralNetworkIO(layer, [1, 2, 'hello'])
        NeuralNetworkIO(layer, {'a' : 0, 'b' : 1})

    with db as session :
        nn = session.access.neural_network('bar')
        ut.assert_equal([io.object for io in nn._structure.input().objects],
                        [None, '', 'baz', session.access.word('foo'), [1, 2, 'hello'], {'a' : 0, 'b' : 1}])

def __test__ (ut) :
    _test_document(ut)
    _test_neural_network_io(ut)

    from nlplib.core.model import Database, Layer

    from sqlalchemy import inspect
    from sqlalchemy import BLOB


    from nlplib.core.model.sqlalchemy_.map import default_mapped
    from nlplib.general import timing

    size = 10000

    db = Database()
    with db as session :
        session.add(NeuralNetwork(list(range(size)), list(range(6)), list(range(size)), name='foo'))

    from pprint import pprint

    @timing
    @db
    def foo (session) :
        nn = session.access.nn('foo')
        for input, connection, output in nn._structure :
            input.values
            connection.weights
            output.values

    foo()

##    with db as session :
##        nn = session.access.nn('foo')

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

