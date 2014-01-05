''' This package contains the database model classes, and various persistence back-ends for these models. '''


from nlplib.core.model.base import Model, SessionDependent

from nlplib.core.model.naturallanguage import Document, Seq, Gram, Word, Index
from nlplib.core.model.neuralnetwork import NeuralNetwork, Perceptron, NeuralNetworkElement, Link, Node, IONode

try :
    from nlplib.core.model.sqlalchemy_ import Database
except ImportError :
    from nlplib.core.model.sqlite3_ import Database

__all__ = ['Model',
           'SessionDependent',

           'Document',
           'Seq',
           'Gram',
           'Word',
           'Index',

           'NeuralNetwork',
           'Perceptron',
           'NeuralNetworkElement',
           'Link',
           'Node',
           'IONode',

           'Database']

