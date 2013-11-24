# todo : a lot
# todo : make <Structure> not session dependent.

from math import tanh

from sqlalchemy.orm.exc import NoResultFound

from nlplib.core.model import Seq, Link, Node, IONode, Word, Access, SessionDependent
from nlplib.general.iter import windowed, chop

def dtanh (y) :
    return 1.0 - y * y

def nodes_at (session, layer) :
    return session._sqlalchemy_session.query(Node).filter_by(layer=layer).all()

def input_nodes_for_seqs (session, seqs) :
    return [session._sqlalchemy_session.query(IONode).filter_by(layer=0, seq_id=seq.id).one()
            for seq in seqs]

class Structure (SessionDependent) :
    def __init__ (self, session, width=10, height=10, *args, **kw) :
        super().__init__(session, *args, **kw)
        self.width  = width
        self.height = height

    def input_node_layer (self) :
        return 0

    def hidden_node_layers (self) :
        return range(self.input_node_layer() + 1, self.output_node_layer())

    def output_node_layer (self) :
        return self.height - 1

    def io_paired_layers (self) :
        size = 2
        return chop(windowed(self.all_layers(), size), size)

    def all_layers (self) :
        return range(self.height)

    def girth (self) :
        return range(self.width)

    def input_nodes (self) :
        return nodes_at(self.session, self.input_node_layer())

    def hidden_nodes (self) :
        for layer in self.hidden_node_layers() :
            yield nodes_at(self.session, layer)

    def output_nodes (self) :
        return nodes_at(self.session, self.output_node_layer())

    def io_paired_nodes (self) :
        for input_layer, output_layer in self.io_paired_layers() :
            # This could be done with far fewer <nodes_at> queries.
            yield (nodes_at(self.session, input_layer), nodes_at(self.session, output_layer))

class Build (SessionDependent) :
    def __init__ (self, *args, **kw) :
        super().__init__(*args, **kw)
        self.structure = Structure(self.session)

    def _get_or_create (self, query, create) :
        result = query()

        if result is None :
            result = create()

        return result

    def _get_or_create_hidden (self, input_nodes, output_nodes) :

        matches = [node for node in self.session._sqlalchemy_session.query(Node).all()
                   if node.input_nodes == input_nodes and node.output_nodes == output_nodes]

        if not matches :
            return self.session.add(Node(layer=1))
        else :
            return matches[0]

    def link_up (self, input_node, output_node, strength) :
        return self._get_or_create(lambda : self.session._sqlalchemy_session.query(Link).get((input_node.id, output_node.id)),
                                   lambda : self.session.add(Link(input_node, output_node, strength)))

    def link_up_layer (self, layer) :
        switch = {0 : 1.0 / 2} # len(input_nodes)
        for input_node in nodes_at(self.session, layer) :
            for output_node in nodes_at(self.session, layer+1) :
                self.link_up(input_node, output_node, switch.get(layer, 0.1))

    def link_up_all (self) :
        for layer in self.structure.all_layers() :
            self.link_up_layer(layer)

    def _get_or_create_io_node (self, seq, layer) :
        try :
            io_node = self.session._sqlalchemy_session.query(IONode).filter_by(layer=layer, seq_id=seq.id).one()
        except NoResultFound :
            io_node = self.session.add(IONode(layer, seq))

        return io_node

    def make_io_nodes (self, seqs, layer) :
        for seq in seqs :
            yield self._get_or_create_io_node(seq, layer)

    def make_hidden_nodes (self) :
        for layer in self.structure.hidden_node_layers() :
            for _ in self.structure.girth() :
                yield self.session.add(Node(layer=layer))

    def __call__ (self, input_seqs, output_seqs) :
        input_nodes  = list(self.make_io_nodes(input_seqs, self.structure.input_node_layer()))
        hidden_nodes = list(self.make_hidden_nodes())
        output_nodes = list(self.make_io_nodes(output_seqs, self.structure.output_node_layer()))

        self.link_up_all()

class FeedForward (SessionDependent) :
    def __init__ (self, *args, **kw) :
        super().__init__(*args, **kw)
        self.structure = Structure(self.session)

    def _get_link (self, input_node, output_node) :
        return self.session._sqlalchemy_session.query(Link).get((input_node.id, output_node.id))

    def activate (self, input_nodes) :
        for input_node in input_nodes :
            input_node.current = 1.0

    def fire (self, input_nodes, output_nodes) :
        for output_node in output_nodes :
            total = sum(input_node.current * self._get_link(input_node, output_node).strength
                        for input_node in input_nodes)
            output_node.current = tanh(total)

    def results (self) :
        access = Access(self.session)
        for output_node in self.structure.output_nodes() :
            yield (access.specific(Seq, output_node.seq_id), output_node.current)

    def __call__ (self, active_input_nodes) :
        self.activate(active_input_nodes)

        self.fire(active_input_nodes, nodes_at(self.session, 1))

        io_paired_nodes = self.structure.io_paired_nodes()
        next(io_paired_nodes) # The first pair is skipped, because it was just done above.
        for input_nodes, output_nodes in io_paired_nodes :
            self.fire(input_nodes, output_nodes)

        return self.results()

class BackPropagate (SessionDependent) :
    def back_propagate(self, targets, n=0.5) :
        input_nodes  = self.nodes(0)
        hidden_nodes = self.nodes(1)
        output_nodes = self.nodes(2)

        output_deltas = [0.0] * len(targets)
        for i, output_node in enumerate(output_nodes) :
            error = targets[i] - output_node.current
            output_deltas[i] = dtanh(output_node.current) * error

        hidden_deltas = [0.0] * len(hidden_nodes)
        for i, hidden_node in enumerate(hidden_nodes) :
            error = sum(output_deltas[j] * self._get_link(hidden_node, output_node).strength
                        for j, output_node in enumerate(output_nodes))
            hidden_deltas[i] = dtanh(hidden_node.current) * error

        for i, hidden_node in enumerate(hidden_nodes) :
            for j, output_node in enumerate(output_nodes) :
                change = output_deltas[j] * hidden_node.current
                self._get_link(hidden_node, output_node).strength += n * change

        for i, input_node in enumerate(input_nodes) :
            for j, hidden_node in enumerate(hidden_nodes) :
                change = hidden_deltas[j] * input_node.current
                try :
                    self._get_link(input_node, hidden_node).strength += n * change
                except :

                    print([node.id for node in input_node.output_nodes],
                          [node.id for node in hidden_node.input_nodes])
                    raise

    def __call__ (self, input_seqs, output_seqs, correct) :

        #self.build(input_seqs, output_seqs)

        self.input(input_seqs)

        targets = [0.0] * len(output_seqs)

        targets[output_seqs.index(correct)] = 1.0

        error = self.back_propagate(targets)

    def input (self, seqs) :
        # redundant
        return self.feed_forward(input_nodes_for_seqs(self.session, seqs))

class NeuralNetwork (SessionDependent) :
    def __init__ (self, *args, **kw) :
        super().__init__(*args, **kw)

        self.feed_forward = FeedForward(self.session)
        self.build        = Build(self.session)
        self.train        = BackPropagate(self.session)



    def input (self, seqs) :
        return self.feed_forward(input_nodes_for_seqs(self.session, seqs))


def __demo__ () :
    from nlplib.core.model import Database

    db = Database()

    with db as session :
        for char in 'abcdef' :
            session.add(Word(char))

    with db as session :
        nn = NeuralNetwork(session)

        access = Access(session)
        global print
        print(0)
        nn.build(access.words('a b'), access.words('d e f'))
        print(1)
        def print (*a, **k) :
            pass

        print(Structure(session).input_nodes())
        for layer in Structure(session).hidden_nodes() :
            print(layer)

        print(Structure(session).output_nodes())
        print()
        for pair in nn.input(access.words('a b')) :
            print(pair)

##        output_seqs = access.words('d e f')
##        for i in range(30) :
##            nn.train(access.words('a c'), output_seqs, output_seqs[0])
##            nn.train(access.words('b c'), output_seqs, output_seqs[1])
##            nn.train(access.words('a'), output_seqs, output_seqs[2])
##            print(i)
##
##        print(nn.input(access.words('a c')))
##        # [0.861, 0.011, 0.016]
##
##        print(nn.input(access.words('b c')))
##        # [-0.030, 0.883, 0.006]
##
##        print(nn.input(access.words('c')))
##        # [0.8459277961565395, -0.011590385221469553, -0.8361964445052618]


def __test__ (ut) :
    structure = Structure(None, 1, 2)
    ut.assert_equal(list(structure.io_paired_layers()),   [(0, 1)] )
    ut.assert_equal(list(structure.all_layers()),         [0, 1]   )
    ut.assert_equal(list(structure.hidden_node_layers()), []       )

    structure = Structure(None, 1, 3)
    ut.assert_equal(list(structure.io_paired_layers()),   [(0, 1), (1, 2)] )
    ut.assert_equal(list(structure.all_layers()),         [0, 1, 2]        )
    ut.assert_equal(list(structure.hidden_node_layers()), [1]              )
    ut.assert_equal(list(structure.girth()),              [0]              )
    ut.assert_equal(structure.output_node_layer(),        2                )
    ut.assert_equal(structure.input_node_layer(),         0                )

    structure = Structure(None, 2, 4)
    ut.assert_equal(list(structure.io_paired_layers()),   [(0, 1), (1, 2), (2, 3)] )
    ut.assert_equal(list(structure.all_layers()),         [0, 1, 2, 3]             )
    ut.assert_equal(list(structure.hidden_node_layers()), [1, 2]                   )
    ut.assert_equal(list(structure.girth()),              [0, 1]                   )
    ut.assert_equal(structure.output_node_layer(),        3                        )
    ut.assert_equal(structure.input_node_layer(),         0                        )

if __name__ == '__main__' :
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())
    __demo__()

