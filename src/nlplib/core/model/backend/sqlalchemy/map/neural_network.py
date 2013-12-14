''' This module outlines how neural network related models are mapped to their respective SQLAlchemy tables. '''


from sqlalchemy.orm import relationship, column_property, foreign
from sqlalchemy.sql import and_, not_
from sqlalchemy import Column, Boolean, Integer, Float, String, ForeignKey

from nlplib.core.model.backend.sqlalchemy.map.base import ClassMapper
from nlplib.core.model.neural_network import NeuralNetwork, NeuralNetworkElement, Link, Node, IONode

class NeuralNetworkMapper (ClassMapper) :
    cls  = NeuralNetwork
    name = 'neural_network'

    def columns (self) :
        return (Column('id', Integer, primary_key=True),
                Column('name', String, unique=True, nullable=False))

    def mapper_kw (self) :
        io_node_cls   = self.classes['io_node']
        io_node_table = self.tables['io_node']
        element_table = self.tables['neural_network_element']

        input_relation = relationship(io_node_cls,
                                      primaryjoin=and_(self.table.c.id==foreign(element_table.c.neural_network_id),
                                                       io_node_table.c.is_input))

        output_relation = relationship(io_node_cls,
                                       primaryjoin=and_(self.table.c.id==foreign(element_table.c.neural_network_id),
                                                        not_(io_node_table.c.is_input)))

        return {'properties' : {'elements'     : relationship(self.classes['neural_network_element'],
                                                              backref='neural_network'),
                                'links'        : relationship(self.classes['link']),
                                'nodes'        : relationship(self.classes['node']),
                                'io_nodes'     : relationship(self.classes['io_node']),
                                'input_nodes'  : input_relation,
                                'output_nodes' : output_relation,
                                '_id'          : self.table.c.id}}

class NeuralNetworkElementMapper (ClassMapper) :
    cls  = NeuralNetworkElement
    name = 'neural_network_element'

    def columns (self) :
        return (Column('id', Integer, primary_key=True),
                Column('type', String),
                Column('neural_network_id', Integer, ForeignKey('neural_network.id'), nullable=False))

    def mapper_kw (self) :
        return {'properties' : {'_id'                : self.table.c.id,
                                '_type'              : self.table.c.type,
                                '_neural_network_id' : self.table.c.neural_network_id},
                'polymorphic_identity' : self.name,
                'polymorphic_on' : self.table.c.type}

class LinkMapper (ClassMapper) :
    cls  = Link
    name = 'link'

    def columns (self) :
        return (Column('id', Integer, ForeignKey('neural_network_element.id'), primary_key=True),
                Column('strength', Float),

                Column('input_node_id', Integer, ForeignKey('node.id'), primary_key=True),
                Column('output_node_id', Integer, ForeignKey('node.id'), primary_key=True))

    def mapper_kw (self) :
        return {'inherits' : self.classes['neural_network_element'],
                'polymorphic_identity' : self.name,
                'properties' : {'input_node'      : relationship(self.classes['node'],
                                                                 foreign_keys=(self.table.c.input_node_id,)),
                                'output_node'     : relationship(self.classes['node'],
                                                                 foreign_keys=(self.table.c.output_node_id,)),
                                '_input_node_id'  : self.table.c.input_node_id,
                                '_output_node_id' : self.table.c.output_node_id,
                                '_id'             : column_property(self.table.c.id,
                                                                    self.tables['neural_network_element'].c.id)}}

class NodeMapper (ClassMapper) :
    cls  = Node
    name = 'node'

    def columns (self) :
        return (Column('id', Integer, ForeignKey('neural_network_element.id'), primary_key=True),
                Column('layer_index', Integer),
                Column('current', Float))

    def mapper_kw (self) :
        link_table = self.tables['link']

        relation = relationship(self.cls,
                                secondary=link_table,
                                primaryjoin=self.table.c.id==link_table.c.input_node_id,
                                secondaryjoin=self.table.c.id==link_table.c.output_node_id,
                                backref='input_nodes')

        return {'properties' : {'output_nodes' : relation,
                                '_id'          : column_property(self.table.c.id,
                                                                 self.tables['neural_network_element'].c.id)},
                'inherits' : self.classes['neural_network_element'],
                'polymorphic_identity' : self.name}

class IONodeMapper (ClassMapper) :
    cls  = IONode
    name = 'io_node'

    def columns (self) :
        return (Column('id', Integer, ForeignKey('node.id'), primary_key=True),
                Column('seq_id', Integer, ForeignKey('seq.id'), nullable=False),
                Column('is_input', Boolean, nullable=False))

    def mapper_kw (self) :
        return {'properties' : {'seq'     : relationship(self.classes['seq']),
                                '_seq_id' : self.table.c.seq_id,
                                '_id'     : column_property(self.table.c.id,
                                                            self.tables['neural_network_element'].c.id,
                                                            self.tables['node'].c.id)},
                'inherits' : self.classes['node'],
                'polymorphic_identity' : self.name}

