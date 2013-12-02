''' This module outlines how neural network related models are mapped to their respective SQLAlchemy tables. '''


from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, Float, String, ForeignKey, Table

from nlplib.core.model.backend.sqlalchemy.map.base import ClassMapper
from nlplib.core.model.neural_network import NeuralNetwork, NeuralNetworkElement, Link, Node, IONode

class NeuralNetworkMapper (ClassMapper) :
    cls  = NeuralNetwork
    name = 'neural_network'

    def columns (self) :
        return (Column('id', Integer, primary_key=True),
                Column('name', String, unique=True, nullable=False))

    def mapper_kw (self) :
        return {'properties' : {'elements' : relationship(self.classes['neural_network_element']),
                                'links'    : relationship(self.classes['link']),
                                'nodes'    : relationship(self.classes['node']),
                                'io_nodes' : relationship(self.classes['io_node'])}}

class NeuralNetworkElementMapper (ClassMapper) :
    cls  = NeuralNetworkElement
    name = 'neural_network_element'

    def columns (self) :
        return (Column('id', Integer, primary_key=True),
                Column('type', String),
                Column('neural_network_id', Integer, ForeignKey('neural_network.id'), nullable=False))

    def mapper_kw (self) :
        return {'polymorphic_identity' : self.name,
                'polymorphic_on' : self.table.c.type}

class LinkMapper (ClassMapper) :
    cls  = Link
    name = 'link'

    def columns (self) :
        return (Column('id', Integer, ForeignKey('neural_network_element.id'), primary_key=True),
                Column('input_node_id', Integer, ForeignKey('node.id'), primary_key=True),
                Column('output_node_id', Integer, ForeignKey('node.id'), primary_key=True),
                Column('strength', Float))

    def mapper_kw (self) :
        return {'inherits' : self.classes['neural_network_element'],
                'polymorphic_identity' : self.name}

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

        return {'inherits' : self.classes['neural_network_element'],
                'polymorphic_identity' : self.name,
                'properties' : {'output_nodes' : relation}}

class IONodeMapper (ClassMapper) :
    cls  = IONode
    name = 'io_node'

    def columns (self) :
        return (Column('id', Integer, ForeignKey('node.id'), primary_key=True),
                Column('seq_id', Integer, ForeignKey('seq.id'), nullable=False))

    def mapper_kw (self) :
        return {'inherits' : self.classes['node'],
                'polymorphic_identity' : self.name}

