

from nlplib.core.model import SessionDependent

__all__ = ['NeuralNetworkDependent']

class NeuralNetworkDependent (SessionDependent) :
    def __init__ (self, session, neural_network, *args, **kw) :
        super().__init__(session, *args, **kw)
        self.neural_network = neural_network


