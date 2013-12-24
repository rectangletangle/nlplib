

from sqlalchemy.orm import mapper
from sqlalchemy import Table

from nlplib.core.base import Base

__all__ = ['ClassMapper']

class ClassMapper (Base) :
    ''' This acts as an abstract base class for mapping model classes to SQLAlchemy table objects.

        Note : This modifies some of the model class's (<ClassMapper.cls>) attributes. '''

    # Classical style SQLAlchemy mapping is used, because although SQLAlchemy's declarative tools are awesome, they
    # would severely mess up the inheritance model by requiring all of the models to inherit from SQLAlchemy's
    # declarative base.

    cls   = None
    name  = None

    def __init__ (self, metadata, tables, classes) :
        self.metadata = metadata
        self.tables   = tables
        self.classes  = classes

        self.table = Table(self.name, self.metadata, *self.columns())

    def map (self) :
        return mapper(self.cls, self.table, **self.mapper_kw())

    def columns (self) :
        return ()

    def mapper_kw (self) :
        return {}

