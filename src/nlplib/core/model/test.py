''' This tests the models. '''

import random

from nlplib.core.process.index import Indexed
from nlplib.core.model.exc import StorageError
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
        append(NeuralNetworkIO(structure, b'\xc3\x9c'.decode()))
        append(NeuralNetworkIO(structure, session.access.word('foo')))
        append(NeuralNetworkIO(structure, [1, 2, 'hello']))
        append(NeuralNetworkIO(structure, {'a' : 0, 'b' : 1, 2 : 'c'})) # The <2> key is converted into a string.

    with db as session :
        nn = session.access.neural_network('bar')
        ut.assert_equal(list(nn._structure.inputs().objects()),
                        [None, '', 'baz', b'\xc3\x9c'.decode(), session.access.word('foo'), [1, 2, 'hello'],
                         {'a' : 0, 'b' : 1, '2' : 'c'}])

def _test_neural_network_methods (ut) :
    # Neural network connection weights are initiated pseudorandomly, this introduces determinism to the randomness.
    random.seed(0)

    nn = NeuralNetwork('abc', 3, 'def', name='foo')

    training_patterns = [('ab', 'f'),
                         ('ac', 'f'),
                         ('b',  'd'),
                         ('c',  'e')]

    error = round(sum(nn.train(inputs, outputs) for _ in range(20) for inputs, outputs in training_patterns), 5)

    ut.assert_equal(error, 10.51832)

    correct_outputs = [('f', 'd', 'e'),
                       ('d', 'e', 'f'),
                       ('e', 'd', 'f'),
                       ('f', 'd', 'e'),
                       ('f', 'e', 'd'),
                       ('d', 'e', 'f')]

    for inputs, correct in zip(['a', 'b', 'c', 'ab', 'ac', 'bc'], correct_outputs) :
        ut.assert_equal(tuple(object for object, score in sorted(nn.predict(inputs), reverse=True)), correct)

    def tests (nn) :
        ut.assert_true('a' in nn)
        ut.assert_true('d' in nn)
        ut.assert_true('z' not in nn)
        ut.assert_true(173 not in nn)

        ut.assert_equal(list(nn.inputs()), list('abc'))
        ut.assert_equal(list(nn.outputs()), list('def'))
        ut.assert_equal(list(nn), list('abcdef'))
        ut.assert_equal([(object, '%.5f' % score) for object, score in nn.scores()],
                        [('d', '0.68689'), ('e', '0.60615'), ('f', '-0.05136')])

    tests(nn)

    db = Database()

    with db as session :
        session.add(nn)

    tests(nn)

    with db as session :
        tests(session.access.nn('foo'))

    tests(nn)

def _test_neural_network_names (ut) :

    db = Database()

    with db as session :
        session.add(NeuralNetwork(3, 3))
        session.add(NeuralNetwork(4, 4))
        session.add(NeuralNetwork(4, 4, name=['a', 'b']))
        session.add(NeuralNetwork(4, 4, name='a'))
        session.add(NeuralNetwork(4, 4, name=1))
        session.add(NeuralNetwork(4, 4, name={'a' : 0, 1 : 'b'}))

    @db
    def add_non_unique_name (session) :
         session.add(NeuralNetwork(4, 4, name='a'))

    ut.assert_raises(add_non_unique_name, StorageError)

    @db
    def add_complex_non_unique_name (session) :
         session.add(NeuralNetwork(4, 4, name=['a', 'b']))

    ut.assert_raises(add_complex_non_unique_name, StorageError)

    with db as session :
        ut.assert_equal(len(list(session.access.all_neural_networks())), 6)

        nn_a = session.access.nn('a')
        nn_a_again = session.access.nn('a')
        ut.assert_true(nn_a is not None)
        ut.assert_true(nn_a is nn_a_again)

        nn_a_b = session.access.nn(['a', 'b'])
        ut.assert_true(nn_a_b is not None)

        nn_none = session.access.nn(None)
        ut.assert_true(nn_none is not None)

        nn_one = session.access.neural_network(1)
        ut.assert_true(nn_one is not None)

        dict_nn = session.access.neural_network({'a' : 0, 1 : 'b'})
        ut.assert_equal(dict_nn.name, {'a' : 0, '1' : 'b'})

        ut.assert_equal(len({id(nn) for nn in [nn_a, nn_a_b, nn_none, nn_one, dict_nn]}), 5)

    db = Database()

    with db as session :
        session.add(NeuralNetwork(4, 4, name='a'))
        session.add(NeuralNetwork(4, 4, name='b'))

    with db as session :
        ut.assert_true(session.access.nn(None) is None)

def __test__ (ut) :
    _test_document(ut)
    _test_neural_network_io(ut)
    _test_neural_network_methods(ut)
    _test_neural_network_names(ut)

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

