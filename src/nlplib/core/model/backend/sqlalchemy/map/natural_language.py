''' This module outlines how natural language related models are mapped to their respective SQLAlchemy tables. '''


from sqlalchemy.orm import relationship, reconstructor, backref
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, UniqueConstraint, Table

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
        association = Table('document_seq_association',
                            self.metadata,
                            Column('_document_id', Integer, ForeignKey('document.id')),
                            Column('_seq_id', Integer, ForeignKey('seq.id')))

        return {'properties' : {'seqs' : relationship(self.classes['seq'], secondary=association)}}

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
        reconstructor(self.cls._make_seqs)

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
                Column('first_token', Integer),
                Column('last_token', Integer),
                Column('first_character', Integer),
                Column('last_character', Integer),
                Column('tokenization_algorithm', String),

                Column('_document_id', Integer, ForeignKey('document.id'), nullable=False),
                Column('_seq_id', Integer, ForeignKey('seq.id'), nullable=False))

    def mapper_kw (self) :
        return {'properties' : {'document' : relationship(self.classes['document'])}}
    # 'seq'      : relationship(self.classes['seq'], backref='indexes')}


