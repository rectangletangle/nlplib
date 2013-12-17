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
        ut.assert_equal([link.affinity for link in nn_a.links], [0, 1])
        ut.assert_equal(len(nn_a.nodes), 4)
        ut.assert_equal([io_node.seq for io_node in nn_a.io_nodes], [Word('a'), Word('b')])

        ut.assert_equal(len(nn_b.elements), 3)
        ut.assert_equal([link.affinity for link in nn_b.links], [2])
        ut.assert_equal(len(nn_b.nodes), 2)
        ut.assert_equal([io_node.seq for io_node in nn_b.io_nodes], [Word('c')])

        ut.assert_equal(len(nn_c.elements), 0 )
        ut.assert_equal(len(nn_c.links),    0 )
        ut.assert_equal(len(nn_c.nodes),    0 )
        ut.assert_equal(len(nn_c.io_nodes), 0 )

def _test_neural_network_links (ut) :

    db = Database()

    with db as session :
        session.add(NeuralNetwork('foo'))
        session.add(Word('bar'))

    with db as session :
        nn = session.access.neural_network('foo')
        word = session.access.word('bar')

        nn.nodes.extend([IONode(nn, seq=word, is_input=True), Node(nn)])

    with db as session :
        nn = session.access.neural_network('foo')
        io_node, node = nn.nodes

        link = session.add(Link(nn, io_node, node))

        ut.assert_true(io_node.output_nodes[node] is node.input_nodes[io_node] is link)
        ut.assert_equal(io_node.input_nodes, {})
        ut.assert_equal(node.output_nodes, {})
        ut.assert_equal(io_node.output_nodes, {node : link})
        ut.assert_equal(node.input_nodes, {io_node : link})

def __test__ (ut) :
    _test_neural_network_model(ut)
    _test_neural_network_links(ut)

if __name__ == '__main__' :
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())

