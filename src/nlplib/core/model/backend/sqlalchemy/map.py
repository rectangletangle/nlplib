

from collections import OrderedDict

from sqlalchemy.orm import relationship, mapper, reconstructor
from sqlalchemy.exc import ArgumentError
from sqlalchemy import Table, Column, Integer, Float, String, DateTime, Text, ForeignKey, MetaData

from nlplib.core.model import Document, Seq, Gram, Word, Index, Link, Node, IONode
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

    Table('link',
          metadata,
          Column('input_node_id', Integer, ForeignKey('node.id'), primary_key=True),
          Column('output_node_id', Integer, ForeignKey('node.id'), primary_key=True),
          Column('strength', Float))

    Table('node',
          metadata,
          Column('id', Integer, primary_key=True),
          Column('type', String),
          Column('layer', Integer),
          Column('current', Float))

    Table('io_node',
          metadata,
          Column('id', Integer, ForeignKey('node.id'), primary_key=True),
          Column('seq_id', Integer, ForeignKey('seq.id'), nullable=False))

    tables = metadata.tables

    def __init__ (self, document_cls=Document, seq_cls=Seq, gram_cls=Gram, word_cls=Word, index_cls=Index,
                  link_cls=Link, node_cls=Node, io_node_cls=IONode) :

        self.classes = classes = OrderedDict()
        classes['document'] = document_cls
        classes['seq']      = seq_cls
        classes['gram']     = gram_cls
        classes['word']     = word_cls
        classes['index']    = index_cls
        classes['link']     = link_cls
        classes['node']     = node_cls
        classes['io_node']  = io_node_cls

    def __call__ (self) :
        ''' This maps classes to their respective tables. '''

        for name, cls in self.classes.items() :
            table = self.tables[name]

            map_cls = self.mappers.get(name, lambda _, cls, table : mapper(cls, table))

            map_cls(self, cls, table)

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

    def _map_node (self, node_cls, node_table) :
        link_table = self.tables['link']

        relation = relationship(node_cls,
                                secondary=link_table,
                                primaryjoin=node_table.c.id==link_table.c.input_node_id,
                                secondaryjoin=node_table.c.id==link_table.c.output_node_id,
                                backref='input_nodes')

        mapper(node_cls,
               node_table,
               polymorphic_identity='node',
               polymorphic_on=node_table.c.type,
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
               'node'     : _map_node,
               'io_node'  : _map_io_node}

try :
    Mapper()()
except ArgumentError :
    pass

