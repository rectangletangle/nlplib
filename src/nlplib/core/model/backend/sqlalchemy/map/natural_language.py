''' This module outlines how natural language related models are mapped to their respective SQLAlchemy tables. '''


from sqlalchemy.orm import relationship, reconstructor, backref
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, UniqueConstraint

from nlplib.core.model.backend.sqlalchemy.map.base import ClassMapper
from nlplib.core.model.natural_language import Document, Seq, Gram, Word, Index

class DocumentMapper (ClassMapper) :
    cls  = Document
    name = 'document'

    def columns (self) :
        return (Column('id', Integer, primary_key=True),
                Column('string', Text),
                Column('length', Integer),
                Column('word_count', Integer),
                Column('title', Text),
                Column('url', String),
                Column('created_on', DateTime))
                # authors  = ?
                # img_urls = Column(String)

    def mapper_kw (self) :
        return {'properties' : {'indexes' : relationship(self.classes['index'])}}

class SeqMapper (ClassMapper) :
    cls  = Seq
    name = 'seq'

    def columns (self) :
        return (Column('id', Integer, primary_key=True),
                Column('type', String),
                Column('string', String, nullable=False),
                Column('prevalence', Integer),
                UniqueConstraint('type', 'string'))

    def mapper_kw (self) :
        return {'properties' : {'indexes' : relationship(self.classes['index'])},
                'polymorphic_identity' : self.name,
                'polymorphic_on' : self.table.c.type}

class GramMapper (ClassMapper) :
    cls  = Gram
    name = 'gram'

    def columns (self) :
        return (Column('id', Integer, ForeignKey('seq.id'), primary_key=True),)

    def mapper_kw (self) :
        return {'inherits' : self.classes['seq'],
                'polymorphic_identity' : self.name}

    def map (self) :
        # This makes sure that SQLAlchemy creates a <gram.words> attribute when a gram object is being built (__init__
        # isn't called).
        reconstructor(self.cls._make_words)

        return super().map()

class WordMapper (ClassMapper) :
    cls  = Word
    name = 'word'

    def columns (self) :
        return (Column('id', Integer, ForeignKey('seq.id'), primary_key=True),)

    def mapper_kw (self) :
        return {'inherits' : self.classes['seq'],
                'polymorphic_identity' : self.name}

class IndexMapper (ClassMapper) :
    cls  = Index
    name = 'index'

    def columns (self) :
        return (Column('id', Integer, primary_key=True),
                Column('document_id', Integer, ForeignKey('document.id'), nullable=False),
                Column('seq_id', Integer, ForeignKey('seq.id'), nullable=False),
                Column('first_token', Integer),
                Column('last_token', Integer),
                Column('first_character', Integer),
                Column('last_character', Integer),
                Column('tokenization_algorithm', String))

class TrieNode :
    def __init__ (self, seq, prevalence=None, parent=None) :
        self.seq_id = seq.id

        self.prevalence = prevalence

        try :
            self.parent_id = parent.id
        except AttributeError :
            self.parent_id = None

    def add_child (self, trie_node) :
        self.children.append(trie_node)
        return trie_node

class TrieNodeMapper (ClassMapper) :
    cls  = TrieNode
    name = 'trie_node'

    def columns (self) :
        return (Column('id', Integer, primary_key=True),
                Column('seq_id', Integer, ForeignKey('seq.id')),
                Column('prevalence', Integer),
                Column('parent_id', Integer, ForeignKey('trie_node.id')))

    def mapper_kw (self) :
        return {'properties' : {'seq'      : relationship(self.classes['seq'], backref='trie_nodes'),
                                'children' : relationship(TrieNode,
                                                          backref=backref('parent', remote_side=[self.table.c.id]))}}

