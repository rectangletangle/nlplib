''' This module outlines how neural network related models are mapped to their respective SQLAlchemy tables. '''

from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.orm import relationship, column_property, backref, reconstructor
from sqlalchemy.sql import and_, not_
from sqlalchemy import Column, Boolean, Integer, Float, String, ForeignKey

from nlplib.core.model.sqlalchemy_.base import ClassMapper
from nlplib.core.model.neuralnetwork import NeuralNetwork, Perceptron, NeuralNetworkElement, Link, Node, IONode

class NeuralNetworkMapper (ClassMapper) :
    cls  = NeuralNetwork
    name = 'neural_network'

    def columns (self) :
        return (Column('id', Integer, primary_key=True),
                Column('type', String),
                Column('name', String, unique=True, index=True))

    def mapper_kw (self) :
        io_node_class = self.classes['io_node']
        io_node_table = self.tables['io_node']
        element_table = self.tables['neural_network_element']

        input_relation = relationship(io_node_class,
                                      primaryjoin=and_(self.table.c.id==element_table.c.neural_network_id,
                                                       io_node_table.c.is_input),
                                      foreign_keys=element_table.c.neural_network_id)

        output_relation = relationship(io_node_class,
                                       primaryjoin=and_(self.table.c.id==element_table.c.neural_network_id,
                                                        not_(io_node_table.c.is_input)),
                                       foreign_keys=element_table.c.neural_network_id)

        return {'properties' : {'inputs'  : input_relation,
                                'outputs' : output_relation,
                                '_id'     : self.table.c.id},
                'polymorphic_identity' : self.name,
                'polymorphic_on' : self.table.c.type}

class PerceptronMapper (ClassMapper) :
    cls  = Perceptron
    name = 'perceptron'

    def columns (self) :
        return (Column('id', Integer, ForeignKey('neural_network.id'), primary_key=True),
                Column('hidden_config', String, nullable=False),
                Column('affinities', String, nullable=False),
                Column('charges', String))

    def mapper_kw (self) :
        return {'properties' : {'_id' : column_property(self.table.c.id, self.tables['neural_network'].c.id),
                                '_hidden_config' : self.table.c.hidden_config,
                                '_affinities'    : self.table.c.affinities,
                                '_charges'       : self.table.c.charges},
                'inherits' : self.classes['neural_network'],
                'polymorphic_identity' : self.name}

class NeuralNetworkElementMapper (ClassMapper) :
    cls  = NeuralNetworkElement
    name = 'neural_network_element'

    def columns (self) :
        return (Column('id', Integer, primary_key=True),
                Column('type', String),
                Column('neural_network_id', Integer, ForeignKey('neural_network.id'), nullable=False, index=True))

    def mapper_kw (self) :
        return {'properties' : {'_id'                : self.table.c.id,
                                '_type'              : self.table.c.type,
                                '_neural_network_id' : self.table.c.neural_network_id,
                                'neural_network'     : relationship(self.classes['neural_network'],
                                                                    backref='elements')},
                'polymorphic_identity' : self.name,
                'polymorphic_on' : self.table.c.type}

class LinkMapper (ClassMapper) :
    cls  = Link
    name = 'link'

    def columns (self) :
        return (Column('id', Integer, ForeignKey('neural_network_element.id'), primary_key=True),
                Column('affinity', Float),

                Column('input_node_id', Integer, ForeignKey('node.id'), primary_key=True),
                Column('output_node_id', Integer, ForeignKey('node.id'), primary_key=True))

    def mapper_kw (self) :

        attr_mapped = attribute_mapped_collection

        input_node_relation = relationship(self.classes['node'], foreign_keys=(self.table.c.input_node_id,),
                                           backref=backref('outputs', collection_class=attr_mapped('output_node')))

        output_node_relation = relationship(self.classes['node'], foreign_keys=(self.table.c.output_node_id,),
                                            backref=backref('inputs', collection_class=attr_mapped('input_node')))

        return {'inherits' : self.classes['neural_network_element'],
                'polymorphic_identity' : self.name,
                'properties' : {'input_node'      : input_node_relation,
                                'output_node'     : output_node_relation,
                                '_input_node_id'  : self.table.c.input_node_id,
                                '_output_node_id' : self.table.c.output_node_id,
                                '_id'             : column_property(self.table.c.id,
                                                                    self.tables['neural_network_element'].c.id)}}

class NodeMapper (ClassMapper) :
    cls  = Node
    name = 'node'

    def columns (self) :
        return (Column('id', Integer, ForeignKey('neural_network_element.id'), primary_key=True),
                Column('charge', Float),
                Column('error', Float))

    def mapper_kw (self) :
        return {'properties' : {'_id' : column_property(self.table.c.id,
                                                        self.tables['neural_network_element'].c.id)},
                'inherits' : self.classes['neural_network_element'],
                'polymorphic_identity' : self.name}

class IONodeMapper (ClassMapper) :
    cls  = IONode
    name = 'io_node'

    def columns (self) :
        return (Column('id', Integer, ForeignKey('node.id'), primary_key=True),
                Column('model_id', Integer, ForeignKey('seq.id')),
                Column('serialized_object', String),
                Column('is_input', Boolean, nullable=False))

    def mapper_kw (self) :
        return {'properties' : {'_model'             : relationship(self.classes['seq']),
                                '_model_id'          : self.table.c.model_id,
                                '_serialized_object' : self.table.c.serialized_object,
                                '_id'                : column_property(self.table.c.id,
                                                                       self.tables['neural_network_element'].c.id,
                                                                       self.tables['node'].c.id)},
                'inherits' : self.classes['node'],
                'polymorphic_identity' : self.name}

    def map (self, *args, **kw) :
        reconstructor(self.cls._make_object)
        super().map(*args, **kw)

