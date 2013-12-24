''' This module outlines how natural language related models are mapped to their respective SQLAlchemy tables. '''


from sqlalchemy.orm import relationship, reconstructor, backref, column_property
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, UniqueConstraint, Table

from nlplib.core.model.sqlalchemy.base import ClassMapper
from nlplib.core.model.naturallanguage import Document, Seq, Gram, Word, Index

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

    def mapper_kw (self) :
        association = Table('document_seq_association',
                            self.metadata,
                            Column('document_id', Integer, ForeignKey('document.id')),
                            Column('seq_id', Integer, ForeignKey('seq.id')))

        return {'properties' : {'seqs' : relationship(self.classes['seq'], secondary=association),
                                '_id'  : self.table.c.id}}

class SeqMapper (ClassMapper) :
    cls  = Seq
    name = 'seq'

    def columns (self) :
        return (Column('id', Integer, primary_key=True),
                Column('type', String),
                Column('string', String, nullable=False),
                Column('count', Integer),
                UniqueConstraint('type', 'string'))

    def mapper_kw (self) :
        return {'properties' : {'indexes' : relationship(self.classes['index']),
                                '_id'     : self.table.c.id,
                                '_type'   : self.table.c.type},
                'polymorphic_identity' : self.name,
                'polymorphic_on' : self.table.c.type}

class GramMapper (ClassMapper) :
    cls  = Gram
    name = 'gram'

    def columns (self) :
        return (Column('id', Integer, ForeignKey('seq.id'), primary_key=True),)

    def mapper_kw (self) :
        return {'properties' : {'_id' : column_property(self.table.c.id, self.tables['seq'].c.id)},
                'inherits' : self.classes['seq'],
                'polymorphic_identity' : self.name}

class WordMapper (ClassMapper) :
    cls  = Word
    name = 'word'

    def columns (self) :
        return (Column('id', Integer, ForeignKey('seq.id'), primary_key=True),)

    def mapper_kw (self) :
        return {'properties' : {'_id' : column_property(self.table.c.id, self.tables['seq'].c.id)},
                'inherits' : self.classes['seq'],
                'polymorphic_identity' : self.name}

class IndexMapper (ClassMapper) :
    cls  = Index
    name = 'index'

    def columns (self) :
        return (Column('id', Integer, primary_key=True),
                Column('first_token', Integer),
                Column('last_token', Integer),
                Column('first_character', Integer),
                Column('last_character', Integer),
                Column('tokenization_algorithm', String),
                Column('document_id', Integer, ForeignKey('document.id'), nullable=False),
                Column('seq_id', Integer, ForeignKey('seq.id'), nullable=False))

    def mapper_kw (self) :
        return {'properties' : {'document'     : relationship(self.classes['document']),
                                '_id'          : self.table.c.id,
                                '_document_id' : self.table.c.document_id,
                                '_seq_id'      : self.table.c.seq_id}}

