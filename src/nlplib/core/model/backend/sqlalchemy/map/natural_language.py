

from sqlalchemy.orm import relationship, reconstructor
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey

from nlplib.core.model.backend.sqlalchemy.map.base import ClassMapper
from nlplib.core.model.natural_language import Document, Seq, Gram, Word, Index

class DocumentMapper (ClassMapper) :
    cls  = Document
    name = 'document'

    def columns (self) :
        return (Column('id', Integer, primary_key=True),
                Column('raw', Text),
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
                Column('raw', String),
                Column('clean', String),
                Column('prevalence', Integer))

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
