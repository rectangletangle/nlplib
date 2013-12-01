

from sqlalchemy.exc import ArgumentError
from sqlalchemy import MetaData

from nlplib.core.model.backend.sqlalchemy.map.base import ClassMapper

# These modules must be imported so that the call to <_all_subclasses> works properly.
from nlplib.core.model.backend.sqlalchemy.map import natural_language
from nlplib.core.model.backend.sqlalchemy.map import neural_network

from nlplib.core import Base

__all__ = ['Mapper', 'default_mapper']

def _all_subclasses (cls) :
    for subclass in cls.__subclasses__() :
        yield subclass
        yield from _all_subclasses(subclass)

class Mapper (Base) :
    def __init__ (self) :
        self.metadata = MetaData()

        self.tables = self.metadata.tables

        self.class_mappers = [class_mapper for class_mapper in _all_subclasses(ClassMapper)
                              if getattr(class_mapper, '__abstract__', False) is not True]

        self.classes = {class_mapper.name : class_mapper.cls
                        for class_mapper in self.class_mappers}

        self.mappers = [class_mapper(self.metadata, self.tables, self.classes)
                        for class_mapper in self.class_mappers]

    def map (self) :
        ''' This maps classes to their respective tables. '''

        for mapper in self.mappers :
            try :
                mapper.map()
            except ArgumentError :
                pass

default_mapper = Mapper()
default_mapper.map()

