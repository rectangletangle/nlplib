''' This module outlines how neural network related models are mapped to their respective SQLAlchemy tables. '''


import json
import pickle

from sqlalchemy.orm import relationship, column_property, reconstructor
from sqlalchemy import Column, Integer, Float, String, ForeignKey, Binary, TypeDecorator

from nlplib.core.model.sqlalchemy_.base import ClassMapper
from nlplib.core.model.neuralnetwork import NeuralNetwork, Structure, Element, Layer, Connection, NeuralNetworkIO
from nlplib.core.model.exc import StorageError

try :
    # An attempt is made to import NumPy powered data structures.
    from nlplib.core.control.neuralnetwork.numpy_ import Array, Matrix
except ImportError :
    # Fall back to the slower pure Python versions, if NumPy isn't installed.
    from nlplib.core.control.neuralnetwork import Array, Matrix

def _encoded_in_json_format (value) :
    return json.dumps(value, separators=(',', ':'))

class _JSONType (TypeDecorator) :

    impl = String

    def process_bind_param (self, value, dialect) :
        if value is not None :
            try :
                return _encoded_in_json_format(value)
            except TypeError :
                raise StorageError('Objects in neural network must be nlplib models or serializable using '
                                   '<json.dumps>.')
        else :
            return None

    def process_result_value (self, value, dialect) :
        return json.loads(value) if value is not None else None

class _ArrayType (TypeDecorator) :

    impl = Binary

    def process_bind_param (self, values, dialect) :
        return pickle.dumps(list(values))

    def process_result_value (self, values, dialect) :
        return Array(pickle.loads(values))

class _MatrixType (TypeDecorator) :

    impl = Binary

    def process_bind_param (self, values, dialect) :
        return pickle.dumps(list(values))

    def process_result_value (self, values, dialect) :
        return Matrix(pickle.loads(values))

class NeuralNetworkMapper (ClassMapper) :
    cls  = NeuralNetwork
    name = 'neural_network'

    def columns (self) :
        return (Column('id', Integer, primary_key=True),
                Column('name', _JSONType, unique=True, index=True, nullable=True))

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
        return {'polymorphic_identity' : self.name,
                'polymorphic_on' : self.table.c.type,
                'properties' : {'_id'                : self.table.c.id,
                                '_type'              : self.table.c.type,
                                '_neural_network_id' : self.table.c.neural_network_id,
                                'layers'             : relationship(self.classes['layer']),
                                'connections'        : relationship(self.classes['connection'])}}

class ElementMapper (ClassMapper) :
    cls  = Element
    name = 'element'

    def columns (self) :
        return (Column('id', Integer, primary_key=True),
                Column('type', String),
                Column('structure_id', Integer, ForeignKey('structure.id'), nullable=False, index=True))

    def mapper_kw (self) :
        return {'polymorphic_identity' : self.name,
                'polymorphic_on' : self.table.c.type,
                'properties' : {'_id'           : self.table.c.id,
                                '_type'         : self.table.c.type,
                                '_structure_id' : self.table.c.structure_id,
                                'structure'     : relationship(self.classes['structure'], backref='elements')}}

class LayerMapper (ClassMapper) :
    cls  = Layer
    name = 'layer'

    def columns (self) :
        return (Column('id', Integer, ForeignKey('element.id'), primary_key=True),
                Column('charges', _ArrayType),
                Column('errors', _ArrayType))

    def mapper_kw (self) :
        return {'inherits' : self.classes['element'],
                'polymorphic_identity' : self.name,
                'properties' : {'_id'      : column_property(self.table.c.id, self.tables['element'].c.id),
                                '_charges' : self.table.c.charges,
                                '_errors'  : self.table.c.errors,
                                'io'       : relationship(self.classes['neural_network_io'],
                                                          foreign_keys=self.tables['neural_network_io'].c.layer_id)}}

class ConnectionMapper (ClassMapper) :
    cls  = Connection
    name = 'connection'

    def columns (self) :
        return (Column('id', Integer, ForeignKey('element.id'), primary_key=True),
                Column('weights', _MatrixType))

    def mapper_kw (self) :
        return {'inherits' : self.classes['element'],
                'polymorphic_identity' : self.name,
                'properties' : {'_id'      : column_property(self.table.c.id, self.tables['element'].c.id),
                                '_weights' : self.table.c.weights}}

class NeuralNetworkIOMapper (ClassMapper) :
    cls  = NeuralNetworkIO
    name = 'neural_network_io'

    def columns (self) :
        return (Column('id', Integer, ForeignKey('element.id'), primary_key=True),
                Column('layer_id', Integer, ForeignKey('layer.id')),
                Column('model_id', Integer, ForeignKey('seq.id')),
                Column('json', _JSONType))

    def mapper_kw (self) :
        return {'inherits' : self.classes['element'],
                'polymorphic_identity' : self.name,
                'properties' : {'_id'       : column_property(self.table.c.id, self.tables['element'].c.id),
                                '_layer_id' : self.table.c.layer_id,
                                '_model_id' : self.table.c.model_id,
                                '_model'    : relationship(self.classes['seq']),
                                '_json'     : self.table.c.json}}

    def map (self, *args, **kw) :
        reconstructor(self.cls._make_object)
        super().map(*args, **kw)

