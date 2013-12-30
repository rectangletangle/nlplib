

from sqlalchemy.exc import ArgumentError
from sqlalchemy import MetaData

from nlplib.core.model.sqlalchemy_.base import ClassMapper

# These modules must be imported so that the call to <subclasses> works properly.
from nlplib.core.model.sqlalchemy_ import naturallanguage, neuralnetwork

from nlplib.core.base import Base
from nlplib.general import subclasses

__all__ = ['Mapped', 'default_mapped']

class Mapped (Base) :
    def __init__ (self) :
        self.metadata = MetaData()

        self.tables = self.metadata.tables

        self.class_mappers = [class_mapper for class_mapper in subclasses(ClassMapper)
                              if getattr(class_mapper, '__abstract__', False) is not True]

        self.classes = {class_mapper.name : class_mapper.cls
                        for class_mapper in self.class_mappers}

        self.mappers = [class_mapper(self.metadata, self.tables, self.classes)
                        for class_mapper in self.class_mappers]

        self.classes_with_tables = {mapper.cls : mapper.table for mapper in self.mappers}

    def map (self) :
        ''' This maps classes to their respective tables. '''

        for mapper in self.mappers :
            try :
                mapper.map()
            except ArgumentError :
                pass

default_mapped = Mapped()
default_mapped.map()

