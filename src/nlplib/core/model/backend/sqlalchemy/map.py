

from collections import OrderedDict

from sqlalchemy.orm import relationship, mapper, reconstructor
from sqlalchemy.exc import ArgumentError
from sqlalchemy import Table, Column, Integer, Float, String, DateTime, Text, ForeignKey, MetaData

from nlplib.core.model import Document, Seq, Gram, Word, Index, NeuralNetwork, NeuralNetworkElement, Link, Node, IONode
from nlplib.core import Base

__all__ = ['Mapper']

class Mapper (Base) :
    ''' This maps the model classes to SQLAlchemy table instances. This modifies some attributes of the model
        classes. '''

    # Classical style SQLAlchemy mapping is used, because although SQLAlchemy's declarative tools are awesome, they
    # would severely mess up the inheritance model by requiring all of the models to inherit from SQLAlchemy's
    # declarative base.

    metadata = MetaData()

    Table('document',
          metadata,
          Column('id', Integer, primary_key=True),
          Column('raw', Text),
          Column('length', Integer),
          Column('word_count', Integer),
          Column('title', Text),
          Column('url', String),
          Column('created_on', DateTime))
          # authors  = ?
          # img_urls = Column(String)

    Table('seq',
          metadata,
          Column('id', Integer, primary_key=True),
          Column('type', String),
          Column('raw', String),
          Column('clean', String),
          Column('prevalence', Integer))

    Table('gram',
          metadata,
          Column('id', Integer, ForeignKey('seq.id'), primary_key=True))

    Table('word',
          metadata,
          Column('id', Integer, ForeignKey('seq.id'), primary_key=True))

    Table('index',
          metadata,
          Column('id', Integer, primary_key=True),
          Column('document_id', Integer, ForeignKey('document.id'), nullable=False),
          Column('seq_id', Integer, ForeignKey('seq.id'), nullable=False),
          Column('first_token', Integer),
          Column('last_token', Integer),
          Column('first_character', Integer),
          Column('last_character', Integer),
          Column('tokenization_algorithm', String))

    Table('neural_network',
          metadata,
          Column('id', Integer, primary_key=True),
          Column('name', String, unique=True, nullable=False))

    Table('neural_network_element',
          metadata,
          Column('id', Integer, primary_key=True),
          Column('type', String),
          Column('neural_network_id', Integer, ForeignKey('neural_network.id'), nullable=False))

    Table('link',
          metadata,
          Column('id', Integer, ForeignKey('neural_network_element.id'), primary_key=True),
          Column('input_node_id', Integer, ForeignKey('node.id'), primary_key=True),
          Column('output_node_id', Integer, ForeignKey('node.id'), primary_key=True),
          Column('strength', Float))

    Table('node',
          metadata,
          Column('id', Integer, ForeignKey('neural_network_element.id'), primary_key=True),
          Column('layer', Integer),
          Column('current', Float))

    Table('io_node',
          metadata,
          Column('id', Integer, ForeignKey('node.id'), primary_key=True),
          Column('seq_id', Integer, ForeignKey('seq.id'), nullable=False))

    tables = metadata.tables

    def __init__ (self,
                  document_cls = Document,
                  seq_cls      = Seq,
                  gram_cls     = Gram,
                  word_cls     = Word,
                  index_cls    = Index,

                  neural_network_cls         = NeuralNetwork,
                  neural_network_element_cls = NeuralNetworkElement,
                  link_cls                   = Link,
                  node_cls                   = Node,
                  io_node_cls                = IONode) :

        self.classes = classes = OrderedDict()

        classes['document'] = document_cls
        classes['seq']      = seq_cls
        classes['gram']     = gram_cls
        classes['word']     = word_cls
        classes['index']    = index_cls

        classes['neural_network']         = neural_network_cls
        classes['neural_network_element'] = neural_network_element_cls
        classes['link']                   = link_cls
        classes['node']                   = node_cls
        classes['io_node']                = io_node_cls

    def __call__ (self) :
        ''' This maps classes to their respective tables. '''

        def map_default (_, cls, table) :
            mapper(cls, table)

        for name, cls in self.classes.items() :
            table = self.tables[name]

            map_cls = self.mappers.get(name, map_default)

            try :
                map_cls(self, cls, table)
            except ArgumentError :
                pass

    def _map_document (self, document_cls, document_table) :
        mapper(document_cls,
               document_table,
               properties={'indexes' : relationship(self.classes['index'])})

    def _map_seq (self, seq_cls, seq_table) :
        mapper(seq_cls,
               seq_table,
               properties={'indexes' : relationship(self.classes['index'])},
               polymorphic_identity='seq',
               polymorphic_on=seq_table.c.type)

    def _map_gram (self, gram_cls, gram_table) :
        # This makes sure that SQLAlchemy creates a <gram.words> attribute when a gram object is being built (__init__
        # isn't called).
        reconstructor(gram_cls._make_words)

        mapper(gram_cls,
               gram_table,
               inherits=self.classes['seq'],
               polymorphic_identity='gram')

    def _map_word (self, word_cls, word_table) :
        mapper(word_cls,
               word_table,
               inherits=self.classes['seq'],
               polymorphic_identity='word')

    def _map_neural_network (self, neural_network_cls, neural_network_table) :
        mapper(neural_network_cls,
               neural_network_table,
               properties={'elements' : relationship(self.classes['neural_network_element']),
                           'links'    : relationship(self.classes['link']),
                           'nodes'    : relationship(self.classes['node']),
                           'io_nodes' : relationship(self.classes['io_node'])})

    def _map_neural_network_element (self, neural_network_element_cls, neural_network_element_table) :
        mapper(neural_network_element_cls,
               neural_network_element_table,
               polymorphic_identity='neural_network_element',
               polymorphic_on=neural_network_element_table.c.type)

    def _map_link (self, link_cls, link_table) :
        mapper(link_cls,
               link_table,
               inherits=self.classes['neural_network_element'],
               polymorphic_identity='link')

    def _map_node (self, node_cls, node_table) :
        link_table = self.tables['link']

        relation = relationship(node_cls,
                                secondary=link_table,
                                primaryjoin=node_table.c.id==link_table.c.input_node_id,
                                secondaryjoin=node_table.c.id==link_table.c.output_node_id,
                                backref='input_nodes')

        mapper(node_cls,
               node_table,
               inherits=self.classes['neural_network_element'],
               polymorphic_identity='node',
               properties={'output_nodes' : relation})

    def _map_io_node (self, io_node_cls, io_node_table) :
        mapper(io_node_cls,
               io_node_table,
               inherits=self.classes['node'],
               polymorphic_identity='io_node')

    mappers = {'document' : _map_document,
               'seq'      : _map_seq,
               'gram'     : _map_gram,
               'word'     : _map_word,

               'neural_network'         : _map_neural_network,
               'neural_network_element' : _map_neural_network_element,
               'link'                   : _map_link,
               'node'                   : _map_node,
               'io_node'                : _map_io_node}

Mapper()()

