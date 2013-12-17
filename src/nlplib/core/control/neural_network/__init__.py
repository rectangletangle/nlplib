

from itertools import islice

from nlplib.core.control.neural_network.base import NeuralNetworkDependent
from nlplib.core.score import Score, Scored
from nlplib.general.iter import windowed, chop
from nlplib.general import math

__all__ = ['FeedForward', 'Backpropagate', 'Prediction', 'Train']

class FeedForward (NeuralNetworkDependent) :
    def __init__ (self, session, neural_network, active_input_nodes, *args, **kw) :
        super().__init__(session, neural_network, *args, **kw)
        self.active_input_nodes = active_input_nodes

    def __iter__ (self) :
        for node in self.neural_network.nodes : # ???
            node.charge = 0.0

        active_nodes = list(self._activate(self.active_input_nodes))

        # todo : use nn.__iter__
        while True :
            current_nodes = list(self._fire(active_nodes))

            if not len(current_nodes) :
                break
            else :
                active_nodes = current_nodes

        return iter(active_nodes)

    def _access_link (self, input_node, output_node) :
        return self.session.access.link(self.neural_network, input_node, output_node)

    def _activate (self, input_nodes) :
        for input_node in input_nodes :
            input_node.charge = 1.0
            yield input_node

    def _fire (self, input_nodes) :
        output_nodes = {output_node
                        for input_node in input_nodes
                        for output_node in input_node.output_nodes}

        for output_node in output_nodes :
            total = sum(input_node.charge * self._access_link(input_node, output_node).affinity
                        for input_node in input_nodes)

            output_node.charge = math.tanh(total)

            yield output_node

class Backpropagate (NeuralNetworkDependent) :
    def back_propagate(self, targets, n=0.1) :
        raise DeprecationWarning
        input_nodes  = self.nodes(0)
        hidden_nodes = self.nodes(1)
        output_nodes = self.nodes(2)

        output_deltas = [0.0] * len(targets)
        for i, output_node in enumerate(output_nodes) :
            error = targets[i] - output_node.current
            output_deltas[i] = math.dtanh(output_node.current) * error

        hidden_deltas = [0.0] * len(hidden_nodes)
        for i, hidden_node in enumerate(hidden_nodes) :
            error = sum(output_deltas[j] * self._get_link(hidden_node, output_node).strength
                        for j, output_node in enumerate(output_nodes))
            hidden_deltas[i] = math.dtanh(hidden_node.current) * error

        for i, hidden_node in enumerate(hidden_nodes) :
            for j, output_node in enumerate(output_nodes) :
                change = output_deltas[j] * hidden_node.current
                self._get_link(hidden_node, output_node).strength += n * change

        for i, input_node in enumerate(input_nodes) :
            for j, hidden_node in enumerate(hidden_nodes) :
                change = hidden_deltas[j] * input_node.current
                self._get_link(input_node, hidden_node).strength += n * change

    def __init__ (self, session, neural_network, active_input_nodes, correct_output_nodes, *args, **kw) :
        super().__init__(session, neural_network, *args, **kw)
        self.active_input_nodes = set(active_input_nodes)
        self.correct_output_nodes = set(correct_output_nodes)

    def __call__ (self) :
        list(FeedForward(self.session, self.neural_network, self.active_input_nodes))
        correct = dict(self._correct())
        error = self._backpropagate(correct)
        return error

    def _calculate_output_deltas (self, correct) :
        for output_node in self.neural_network.output_nodes :
            error = correct[output_node] - output_node.current
            yield (output_node, math.dtanh(output_node.current) * error)

    def _calculate_deltas (self, input_nodes, output_nodes, deltas) :

        access_link = self.session.access.link

        for input_node in input_nodes :

            error = sum(deltas[output_node] * access_link(self.neural_network, input_node, output_node).strength
                        for output_node in output_nodes)

            yield (input_node, math.dtanh(input_node.current) * error)

    def _backpropagate (self, correct, n=0.01) :

        output_deltas = dict(self._calculate_output_deltas(correct))

        for output_layer, input_layer in chop(windowed(reversed(self.neural_network), 2), 2) :

            old_deltas = output_deltas
            output_deltas = dict(self._calculate_deltas(input_layer, output_layer, old_deltas))

            for input_node in input_layer :
                for output_node in output_layer :
                    change = old_deltas[output_node] * input_node.current
                    self.session.access.link(self.neural_network, input_node, output_node).strength += n * change

    def _correct (self) :
        for output_node in self.neural_network.output_nodes :
            if output_node in self.correct_output_nodes :
                yield (output_node, 1.0)
            else :
                yield (output_node, 0.0)

class Prediction (NeuralNetworkDependent) :
    def __init__ (self, session, neural_network, input_seqs, *args, **kw) :
        super().__init__(session, neural_network, *args, **kw)

        self.input_seqs = input_seqs

    def __iter__ (self) :
        active_input_nodes = list(self.session.access.nodes_for_seqs(self.neural_network, self.input_seqs))
        for active_ouput_node in FeedForward(self.session, self.neural_network, active_input_nodes) :
             yield Score(object=active_ouput_node.seq, score=active_ouput_node.current)

class Train (NeuralNetworkDependent) :
    pass

def __test__ (ut) :
    from nlplib.core.control.neural_network.layered import MakeLayeredNeuralNetwork, static_io, static
    from nlplib.core.model import Database, NeuralNetwork, Word

    db = Database()

    with db as session :
        session.add(NeuralNetwork('foo'))

        for char in 'abcdef' :
            session.add(Word(char))

    with db as session :
        config = (static_io(session.access.words('a b')),
                  static(3),
                  static_io(session.access.words('d e f')))

        MakeLayeredNeuralNetwork(session, session.access.neural_network('foo'), config)()


    def as_floats (prediction) :
        return sorted(float(score) for score in prediction)

    with db as session :
        nn = session.access.neural_network('foo')
        #ut.assert_equal(as_floats(Prediction(session, nn, session.access.words('a b'))),
        #                [0.07601250837541615, 0.07601250837541615, 0.07601250837541615])


    with db as session :
        config = (static_io(session.access.words('a b c')),
                  static(3),
                  static(3),
                  static_io(session.access.words('d e f')))

        bar = session.add(NeuralNetwork('bar'))

        MakeLayeredNeuralNetwork(session, bar, config)()


    with db as session :
        bar = session.access.neural_network('bar')

        a, b, c, d, e, f = session.access.nodes_for_seqs(bar, session.access.words('a b c d e f'))

        for _ in range(20) : # 30
            for _ in range(2) :
                Backpropagate(session, bar, [a], [d])()
            Backpropagate(session, bar, [b], [e])()
            Backpropagate(session, bar, [c], [f])()

        print('a', sorted(Prediction(session, bar, session.access.words('a'))))
        print('b', sorted(Prediction(session, bar, session.access.words('b'))))
        print('c', sorted(Prediction(session, bar, session.access.words('c'))))

class FeedForward (NeuralNetworkDependent) :
    def __init__ (self, session, neural_network, active_input_nodes, *args, **kw) :
        super().__init__(session, neural_network, *args, **kw)
        self.active_input_nodes = active_input_nodes

    def __iter__ (self) :
        for node in self.neural_network.nodes :
            node.charge = 0.0

        for node in self.active_input_nodes :
            node.charge = 1.0

        for layer in islice(self.neural_network, 1, None) :
            for node in layer :
                node.charge = math.tanh(node.input_strength())

        yield from layer

class Backpropagate (NeuralNetworkDependent) :
    def __init__ (self, session, neural_network, active_input_nodes, correct_output_nodes, *args, **kw) :
        super().__init__(session, neural_network, *args, **kw)
        self.active_input_nodes = set(active_input_nodes)
        self.correct_output_nodes = set(correct_output_nodes)

    def __call__ (self, n=0.1) :
        output_layer = list(FeedForward(self.session, self.neural_network, self.active_input_nodes))

        for node in output_layer :
            if node in self.correct_output_nodes :
                correct_charge = 1.0
            else :
                correct_charge = 0.0

            error  = correct_charge - node.charge
            delta  = math.dtanh(node.charge) * error
            change = delta * node.charge

            for link in node.input_nodes.values() :
                link.affinity += n * change

        for layer in islice(reversed(self.neural_network), 1, None) :
            for node in layer :
                error  = math.dtanh(node.output_strength()) - node.charge
                delta  = math.dtanh(node.charge) * error
                change = delta * node.charge

                for link in node.input_nodes.values() :
                    link.affinity += n * change

def __test__ (ut) :
    from nlplib.core.control.neural_network.layered import MakeLayeredNeuralNetwork, static_io, static
    from nlplib.core.model import Database, NeuralNetwork, Word

    db = Database()

    with db as session :
        session.add(NeuralNetwork('foo'))
        for char in 'abcdef' :
            session.add(Word(char))

    with db as session :
        config = (static_io(session.access.words('a b c')),)
        config += tuple(static(10) for _ in range(1))
        config += (static_io(session.access.words('d e f')),)
        MakeLayeredNeuralNetwork(session, session.access.neural_network('foo'), config)()

    with db as session :
        nn = session.access.neural_network('foo')
        a = list(session.access.nodes_for_seqs(nn, session.access.words('a')))
        d = list(session.access.nodes_for_seqs(nn, session.access.words('d')))

        b = list(session.access.nodes_for_seqs(nn, session.access.words('b')))
        e = list(session.access.nodes_for_seqs(nn, session.access.words('e')))

        for _ in range(20) :
            Backpropagate(session, nn, a, d)()
            Backpropagate(session, nn, b, e)()

        print('a', sorted(FeedForward(session, nn, a), key=lambda n : n.charge, reverse=True))
        print('b', sorted(FeedForward(session, nn, b), key=lambda n : n.charge, reverse=True))

if __name__ == '__main__' :
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())

