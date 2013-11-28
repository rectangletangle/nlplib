''' This tests the models. '''


from nlplib.core.model import Database, Access, Word, NeuralNetwork, Link, Node, IONode

def _test_neural_network_models (ut) :

    db = Database()

    with db as session :
        session.add(NeuralNetwork('a'))
        session.add(NeuralNetwork('b'))
        session.add(NeuralNetwork('c'))

        session.add(Word('a'))
        session.add(Word('b'))
        session.add(Word('c'))

    with db as session :
        access = Access(session)

        nn_a, nn_b, nn_c = access.all_neural_networks()

        def build_nn (nn, word_string, strength) :

            node    = session.add(Node(nn, 1))
            io_node = session.add(IONode(nn, 0, access.word(word_string)))

            try :
                session._sqlalchemy_session.flush() # todo : remove this
            except AttributeError :
                pass

            session.add(Link(nn, node, io_node, strength))

        build_nn(nn_a, 'a', 0)
        build_nn(nn_a, 'b', 1)
        build_nn(nn_b, 'c', 2)

    with db as session :
        access = Access(session)

        nn_a, nn_b, nn_c = access.all_neural_networks()

        ut.assert_equal(len(nn_a.elements), 6)
        ut.assert_equal([link.strength for link in nn_a.links], [0, 1])
        ut.assert_equal(len(nn_a.nodes), 4)
        ut.assert_equal([access.specific(Word, io_node.seq_id) for io_node in nn_a.io_nodes], [Word('a'), Word('b')])

        ut.assert_equal(len(nn_b.elements), 3)
        ut.assert_equal([link.strength for link in nn_b.links], [2])
        ut.assert_equal(len(nn_b.nodes), 2)
        ut.assert_equal([access.specific(Word, io_node.seq_id) for io_node in nn_b.io_nodes], [Word('c')])

        ut.assert_equal(len(nn_c.elements), 0 )
        ut.assert_equal(len(nn_c.links),    0 )
        ut.assert_equal(len(nn_c.nodes),    0 )
        ut.assert_equal(len(nn_c.io_nodes), 0 )

def __test__ (ut) :
    _test_neural_network_models(ut)

if __name__ == '__main__' :
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())

