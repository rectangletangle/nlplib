

from nlplib.general import math

from nlplib.core.control.neural_network.base import NeuralNetworkDependent

__all__ = ['FeedForward', 'BackPropagate', 'ask', 'train']

class FeedForward (NeuralNetworkDependent) :
    def _access_link (self, input_node, output_node) :
        return self.session.access.link(self.neural_network, input_node, output_node)

    def activate (self, input_nodes) :
        for input_node in input_nodes :
            input_node.current = 1.0

        return input_nodes

    def fire (self, input_nodes) :

        output_nodes = {output_node
                        for input_node in input_nodes
                        for output_node in input_node.output_nodes}

        for output_node in output_nodes :
            total = sum(input_node.current * self._access_link(input_node, output_node).strength
                        for input_node in input_nodes)
            output_node.current = math.tanh(total)

        return output_nodes

    def __call__ (self, active_input_nodes) :
        active_input_nodes = list(active_input_nodes)

        active_nodes = self.activate(active_input_nodes)

        while True :
            current_nodes = self.fire(active_nodes)

            if not len(current_nodes) :
                break
            else :
                active_nodes = current_nodes

        return active_nodes

class BackPropagate (NeuralNetworkDependent) :
    def back_propagate(self, targets, n=0.5) :
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
                try :
                    self._get_link(input_node, hidden_node).strength += n * change
                except :


                    raise

    def __call__ (self, input_seqs, correct_output) :
        raise NotImplementedError('not quite done')
        ask(self.session, self.neural_network, input_seqs)

        targets = [0.0] * len(output_seqs)

        targets[output_seqs.index(correct)] = 1.0

        error = self.back_propagate(targets)

        return error

def ask (session, neural_network, seqs) :
    active_input_nodes = session.access.nodes_for_seqs(neural_network, seqs, layer_index=0)

    return [(node, node.current) for node in FeedForward(session, neural_network)(active_input_nodes)]

def train (session, neural_network, input_seqs, output_seqs, correct_output) :
    return BackPropagate(session, neural_network)(input_seqs, output_seqs, correct_output)

def __demo__ () :
    from nlplib.core.control.neural_network.layered import MakeLayeredNeuralNetwork, static_io, static
    from nlplib.core.model import Database, NeuralNetwork, Word

    db = Database()

    with db as session :
        session.add(NeuralNetwork('foo'))

        for char in 'abcdef' :
            session.add(Word(char))

    with db as session :
        config = (static_io(session.access.words('a b')),
                  static(1),
                  static_io(session.access.words('d e f')))

        MakeLayeredNeuralNetwork(session, session.access.neural_network('foo'), config)()

    with db as session :
        print(ask(session, session.access.neural_network('foo'), session.access.words('a b')))

if __name__ == '__main__' :
    __demo__()

