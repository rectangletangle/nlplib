

import itertools

from nlplib.core.control.neuralnetwork import abstract

__all__ = ['Feedforward', 'Backpropagate']

class Feedforward (abstract.Feedforward) :
    ''' A pure Python implementation of the feedforward neural network algorithm. '''

    def __call__ (self) :

        self._set_charges(self.neural_network.inputs, self.inactive)

        self._set_charges(self.input_nodes, self.active)

        for layer in itertools.islice(self.neural_network, 1, None) :
            for node in layer :
                node.charge = self.activation(self._input_strength(node))

        return layer

    def _set_charges (self, nodes, charge) :
        if callable(charge) :
            for node in nodes :
                node.charge = charge(node)
        else :
            for node in nodes :
                node.charge = charge

    def _input_strength (self, node) :
        return sum(input_node.charge * link.affinity for input_node, link in node.inputs.items())

class Backpropagate (abstract.Backpropagate) :
    ''' A pure Python implementation of the backpropagation neural network training algorithm. '''

    def __call__ (self) :
        output_differences = list(self._output_errors())
        self._hidden_errors()
        self._update_link_affinities()

        total_error = sum(0.5 * difference ** 2 for difference in output_differences)
        return total_error

    def _output_errors (self) :
        for output_node in self.neural_network.outputs :
            correct_value = 1.0 if output_node in self.correct_output_nodes else 0.0

            difference = correct_value - output_node.charge
            output_node.error = self.activation_derivative(output_node.charge) * difference

            yield difference

    def _hidden_errors (self) :
        for layer in self.neural_network.hidden(reverse=True) :
            for node in layer :
                node.error = self.activation_derivative(node.charge) * sum(output_node.error * link.affinity
                                                                           for output_node, link
                                                                           in node.outputs.items())

    def _update_link_affinities (self) :
        for layer in itertools.islice(reversed(self.neural_network), 1, None) :
            for node in layer :
                for link in node.outputs.values() :
                    link.affinity += self.rate * link.output_node.error * node.charge

