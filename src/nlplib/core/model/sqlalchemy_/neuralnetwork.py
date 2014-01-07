''' This module outlines how neural network related models are mapped to their respective SQLAlchemy tables. '''

from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.orm import relationship, column_property, backref, reconstructor
from sqlalchemy.sql import and_, not_
from sqlalchemy import Column, Boolean, Integer, Float, String, ForeignKey, PickleType

from nlplib.core.model.sqlalchemy_.base import ClassMapper
from nlplib.core.model.neuralnetwork import NeuralNetwork, Structure, Element, Layer, Connection, NeuralNetworkIO

class NeuralNetworkMapper (ClassMapper) :
    cls  = NeuralNetwork
    name = 'neural_network'

    def columns (self) :
        return (Column('id', Integer, primary_key=True),
                Column('name', String, unique=True, index=True, nullable=True))

    def mapper_kw (self) :
        return {'properties' : {'_id'        : self.table.c.id,
                                '_structure' : relationship(self.classes['structure'], uselist=False)}}

class StructureMapper (ClassMapper) :
    cls  = Structure
    name = 'structure'

    def columns (self) :
        return (Column('id', Integer, primary_key=True),
                Column('type', String),
                Column('neural_network_id', Integer, ForeignKey('neural_network.id'), nullable=False, index=True))

    def mapper_kw (self) :
        return {'properties' : {'_id'                : self.table.c.id,
                                '_type'              : self.table.c.type,
                                '_neural_network_id' : self.table.c.neural_network_id,
                                'layers'             : relationship(self.classes['layer']),
                                'connections'        : relationship(self.classes['connection'])},
                'polymorphic_identity' : self.name,
                'polymorphic_on' : self.table.c.type}

class ElementMapper (ClassMapper) :
    cls  = Element
    name = 'element'

    def columns (self) :
        return (Column('id', Integer, primary_key=True),
                Column('type', String),
                Column('structure_id', Integer, ForeignKey('structure.id'), nullable=False, index=True))

    def mapper_kw (self) :
        return {'properties' : {'_id'           : self.table.c.id,
                                '_type'         : self.table.c.type,
                                '_structure_id' : self.table.c.structure_id,
                                'structure'     : relationship(self.classes['structure'], backref='elements')},
                'polymorphic_identity' : self.name,
                'polymorphic_on' : self.table.c.type}

class LayerMapper (ClassMapper) :
    cls  = Layer
    name = 'layer'

    def columns (self) :
        return (Column('id', Integer, ForeignKey('element.id'), primary_key=True),
                Column('values', PickleType),
                Column('errors', PickleType))

    def mapper_kw (self) :
        return {'inherits' : self.classes['element'],
                'polymorphic_identity' : self.name,
                'properties' : {'_id' : column_property(self.table.c.id, self.tables['element'].c.id),}}

class ConnectionMapper (ClassMapper) :
    cls  = Connection
    name = 'connection'

    def columns (self) :
        return (Column('id', Integer, ForeignKey('element.id'), primary_key=True),
                Column('weights', PickleType))

    def mapper_kw (self) :
        return {'inherits' : self.classes['element'],
                'polymorphic_identity' : self.name,
                'properties' : {'_id' : column_property(self.table.c.id, self.tables['element'].c.id),}}

class NeuralNetworkIOMapper (ClassMapper) :
    cls  = NeuralNetworkIO
    name = 'neural_network_io'

    def columns (self) :
        return (Column('id', Integer, ForeignKey('element.id'), primary_key=True),
                Column('layer_id', Integer, ForeignKey('layer.id')),
                Column('model_id', Integer, ForeignKey('seq.id')),
                Column('pickled', PickleType))

    def mapper_kw (self) :
        layer_relation = relationship(self.classes['layer'], foreign_keys=self.table.c.layer_id, backref='objects')

        return {'properties' : {'_id'       : column_property(self.table.c.id, self.tables['element'].c.id),
                                '_layer_id' : self.table.c.layer_id,
                                '_model_id' : self.table.c.model_id,
                                '_model'    : relationship(self.classes['seq']),
                                '_pickled'  : self.table.c.pickled,
                                'layer'     : layer_relation},
                'inherits' : self.classes['element'],
                'polymorphic_identity' : self.name}

    def map (self, *args, **kw) :
        reconstructor(self.cls._make_object)
        super().map(*args, **kw)

