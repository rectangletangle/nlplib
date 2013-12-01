''' This tests the models. '''


from nlplib.core.model import Database, Access, Word, NeuralNetwork, Link, Node, IONode, Seq, StaticIOLayer

def _test_neural_network_model (ut) :

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

        def make_nn (nn, word_string, strength) :

            node    = session.add(Node(nn, layer_index=1))
            io_node = session.add(IONode(nn, access.word(word_string), layer_index=0))

            try :
                session._sqlalchemy_session.flush() # todo : remove if possible
            except AttributeError :
                pass

            session.add(Link(nn, node, io_node, strength))

        make_nn(nn_a, 'a', 0)
        make_nn(nn_a, 'b', 1)
        make_nn(nn_b, 'c', 2)

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

def _test_neural_network_links (ut) :
    def get_ids (nodes) :
        return [node.id for node in nodes]

    db = Database()

    with db as session :
        session.add(NeuralNetwork('foo'))
        session.add(Word('bar'))

    with db as session :
        access = Access(session)
        nn = access.neural_network('foo')
        word = access.word('bar')

        nodes = [IONode(nn, seq=word, layer_index=0),
                 IONode(nn, seq=word, layer_index=0),

                 Node(nn, layer_index=1),
                 Node(nn, layer_index=1),

                 IONode(nn, seq=word, layer_index=2),
                 IONode(nn, seq=word, layer_index=2)]

        for node in nodes :
            session.add(node)

        try :
            session._sqlalchemy_session.flush() # todo : remove if possible
        except AttributeError :
            pass

        ids = get_ids(nodes)

    with db as session :
        access = Access(session)
        nn = access.neural_network('foo')
        nodes = [access.specific(Node, id) for id in ids]

        session.add(Link(nn, nodes[0], nodes[2]))
        session.add(Link(nn, nodes[0], nodes[3]))

        session.add(Link(nn, nodes[1], nodes[3]))

        session.add(Link(nn, nodes[3], nodes[4]))
        session.add(Link(nn, nodes[3], nodes[5]))

        ut.assert_equal(nodes[0].input_nodes, [])
        ut.assert_equal(get_ids(nodes[0].output_nodes), [ids[2], ids[3]])
        ut.assert_equal(nodes[1].input_nodes, [])
        ut.assert_equal(get_ids(nodes[1].output_nodes), [ids[3]])
        ut.assert_equal(get_ids(nodes[2].input_nodes), [ids[0]])
        ut.assert_equal(nodes[2].output_nodes, [])
        ut.assert_equal(get_ids(nodes[3].input_nodes), [ids[0], ids[1]])
        ut.assert_equal(get_ids(nodes[3].output_nodes), [ids[4], ids[5]])
        ut.assert_equal(get_ids(nodes[4].input_nodes), [ids[3]])
        ut.assert_equal(nodes[4].output_nodes, [])
        ut.assert_equal(get_ids(nodes[5].input_nodes), [ids[3]])
        ut.assert_equal(nodes[5].output_nodes, [])

def _test_neural_network_layer_configurations (ut) :

    db = Database()

    with db as session :
        for cls, char in zip((Word, Seq, Word), 'abc') :
            session.add(cls(char))

    with db as session :
        session.add(StaticIOLayer(Access(session).all_seqs()))

    with db as session :
        static_io_layer = session._sqlalchemy_session.query(StaticIOLayer).one()

        ut.assert_equal(static_io_layer.seqs, [Word('a'), Seq('b'), Word('c')])

def __test__ (ut) :
    _test_neural_network_model(ut)
    _test_neural_network_links(ut)
    _test_neural_network_layer_configurations(ut)

if __name__ == '__main__' :
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())

