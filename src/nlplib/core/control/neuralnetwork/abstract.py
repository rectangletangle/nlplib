''' This module contains abstract base classes for the various neural network algorithms. '''


from nlplib.core.base import Base
from nlplib.general import math

__all__ = ['Feedforward', 'Backpropagate']

class Feedforward (Base) :
    ''' An abstract base class for implementations of the feed forward neural network algorithm. '''

    def __init__ (self, neural_network, input_nodes, active=1.0, inactive=0.0, activation=math.tanh) :

        self.neural_network = neural_network

        self.input_nodes = input_nodes

        self.active   = active
        self.inactive = inactive

        self.activation = activation

    def __call__ (self) :
        raise NotImplementedError

class Backpropagate (Base) :
    ''' An abstract base class for implementations of the backpropagation neural network training algorithm. '''

    def __init__ (self, neural_network, input_nodes, output_nodes, rate=0.2,
                  activation_derivative=math.dtanh) :

        self.neural_network = neural_network

        self.input_nodes  = set(input_nodes)
        self.output_nodes = set(output_nodes)

        self.rate = rate
        self.activation_derivative = activation_derivative

    def __call__ (self) :
        raise NotImplementedError

