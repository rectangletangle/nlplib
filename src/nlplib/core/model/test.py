''' This tests the models. '''


from nlplib.core.model import Database, Word, NeuralNetwork, Link, Node, IONode, Seq

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
        nn_a, nn_b, nn_c = session.access.all_neural_networks()

        def make_nn (nn, word_string, strength) :

            node    = session.add(Node(nn, layer_index=1))
            io_node = session.add(IONode(nn, session.access.word(word_string), layer_index=0))

            session.add(Link(nn, node, io_node, strength))

        make_nn(nn_a, 'a', 0)
        make_nn(nn_a, 'b', 1)
        make_nn(nn_b, 'c', 2)

    with db as session :
        nn_a, nn_b, nn_c = session.access.all_neural_networks()

        ut.assert_equal(len(nn_a.elements), 6)
        ut.assert_equal([link.strength for link in nn_a.links], [0, 1])
        ut.assert_equal(len(nn_a.nodes), 4)
        ut.assert_equal([session.access.specific(Word, io_node.seq.id) for io_node in nn_a.io_nodes],
                        [Word('a'), Word('b')])

        ut.assert_equal(len(nn_b.elements), 3)
        ut.assert_equal([link.strength for link in nn_b.links], [2])
        ut.assert_equal(len(nn_b.nodes), 2)
        ut.assert_equal([session.access.specific(Word, io_node.seq.id) for io_node in nn_b.io_nodes], [Word('c')])

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
        nn = session.access.neural_network('foo')
        word = session.access.word('bar')

        nodes = [IONode(nn, seq=word, layer_index=0),
                 IONode(nn, seq=word, layer_index=0),

                 Node(nn, layer_index=1),
                 Node(nn, layer_index=1),

                 IONode(nn, seq=word, layer_index=2),
                 IONode(nn, seq=word, layer_index=2)]

        nn.nodes.extend(nodes)

    with db as session :
        nn = session.access.neural_network('foo')
        nodes = nn.nodes

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

def __test__ (ut) :
    _test_neural_network_model(ut)
    #_test_neural_network_links(ut) todo : re implement

if __name__ == '__main__' :
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())

