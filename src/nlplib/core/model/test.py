''' This tests the models. '''

from nlplib.core.process.index import Indexed
from nlplib.core.model import Database, Document, Word, NeuralNetwork, Link, Node, IONode, Seq

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
        ut.assert_equal(len(list(session.access.all_documents())), 3)

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

def _test_neural_network_structure (ut) :

    db = Database()

    with db as session :
        session.add(NeuralNetwork('a'))
        session.add(NeuralNetwork('b'))
        session.add(NeuralNetwork('c'))

        session.add(Word('a'))
        session.add(Word('b'))
        session.add(Word('c'))

    with db as session :
        nn_a, nn_b, nn_c = session.access.all_neural_networks()

        def make_nn (nn, word_string, affinity) :

            node    = session.add(Node(nn))
            io_node = session.add(IONode(nn, session.access.word(word_string), is_input=True))

            session.add(Link(nn, node, io_node, affinity))

        make_nn(nn_a, 'a', 0)
        make_nn(nn_a, 'b', 1)
        make_nn(nn_b, 'c', 2)

    with db as session :
        nn_a, nn_b, nn_c = session.access.all_neural_networks()

        ut.assert_equal(len(nn_a.elements), 6)
        ut.assert_equal([io_node.object for io_node in nn_a.inputs], [Word('a'), Word('b')])

        ut.assert_equal(len(nn_b.elements), 3)
        ut.assert_equal([io_node.object for io_node in nn_b.inputs], [Word('c')])

        ut.assert_equal(len(nn_c.elements), 0)
        ut.assert_equal(len(nn_c.inputs), 0)
        ut.assert_equal(len(nn_c.outputs), 0)

def _test_neural_network_links (ut) :

    db = Database()

    with db as session :
        session.add(NeuralNetwork('foo'))
        session.add(Word('bar'))

    with db as session :
        nn = session.access.neural_network('foo')
        word = session.access.word('bar')

        nn.elements.extend([IONode(nn, word, is_input=True), Node(nn)])

    with db as session :
        nn = session.access.neural_network('foo')
        io_node, node = nn.elements

        link = session.add(Link(nn, io_node, node))

        ut.assert_true(io_node.outputs[node] is node.inputs[io_node] is link)
        ut.assert_equal(io_node.inputs, {})
        ut.assert_equal(node.outputs, {})
        ut.assert_equal(io_node.outputs, {node : link})
        ut.assert_equal(node.inputs, {io_node : link})

def _test_neural_network (ut) :
    # Tests the addition and removal of neural networks and associated objects from the database.

    db = Database()

    with db as session :
        for name in ['a', 'b'] :
            nn = session.add(NeuralNetwork(name))
            Link(nn, IONode(nn, None, True), Node(nn, None))

    def test_counts (session, io, node, link) :
        ut.assert_equal(len(list(session.access.all_io_nodes())), io)
        ut.assert_equal(len(list(session.access.all_nodes())), node)
        ut.assert_equal(len(list(session.access.all_links())), link)

    with db as session :
        test_counts(session, 2, 4, 2)

        session.remove(session.access.nn('a'))

    with db as session :
        test_counts(session, 1, 2, 1)

        all_neural_networks = list(session.access.all_neural_networks())

        ut.assert_equal(len(all_neural_networks), 1)
        ut.assert_equal(all_neural_networks[0].name, 'b')

        session.remove(session.access.nn('b'))

        test_counts(session, 0, 0, 0)

    with db as session :
        test_counts(session, 0, 0, 0)

def _test_io_node (ut) :
    # IO nodes should be able to handle sequences, <None>, or pickle-able Python objects as their object property.

    db = Database()

    with db as session :
        session.add(Word('foo'))

    with db as session :
        nn = session.add(NeuralNetwork('bar'))
        IONode(nn, None, True)
        IONode(nn, '', True)
        IONode(nn, 'baz', True)
        IONode(nn, session.access.word('foo'), True)
        IONode(nn, [1, 2, 'hello'], True)
        IONode(nn, {'a' : 0, 'b' : 1}, True)

    with db as session :
        nn = session.access.neural_network('bar')
        ut.assert_equal([io_node.object for io_node in nn.inputs],
                        [None, '', 'baz', session.access.word('foo'), [1, 2, 'hello'], {'a' : 0, 'b' : 1}])

def __test__ (ut) :
    _test_document(ut)

    _test_neural_network_structure(ut)
    _test_neural_network_links(ut)
    _test_neural_network(ut)
    _test_io_node(ut)

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

