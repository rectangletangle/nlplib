

from sqlalchemy.sql import or_
from sqlalchemy import func

from nlplib.core.model.abstract import access as abstract
from nlplib.core.model import Document, Seq, Gram, Word, Index, NeuralNetwork, Node, IONode, Link
from nlplib.general.iterate import chunked

__all__ = ['Access']

class Access (abstract.Access) :
    def _all (self, cls, chunk_size=100) :
        yield from self.session._sqlalchemy_session.query(cls).yield_per(chunk_size)

    def _seq (self, cls, string) :
        return self.session._sqlalchemy_session.query(cls).filter_by(string=string).first()

    def specific (self, cls, id) :
        return self.session._sqlalchemy_session.query(cls).get(id)

    def most_common (self, cls=Seq, top=10) :

        session = self.session._sqlalchemy_session
        query = session.query(cls).join(Index).group_by(cls).order_by(func.count(Index._seq_id).desc())

        return query.slice(0, top).all()

    def indexes (self, document) :
        return self.session._sqlalchemy_session.query(Index, Seq).filter(Index.document == document).join(Seq).all()

    def matching (self, strings, cls=Seq, chunk_size=100) :
        for chunked_strings in chunked(strings, chunk_size, trail=True) :
            for match in self.session._sqlalchemy_session.query(cls).filter(cls.string.in_(chunked_strings)).all() :
                yield match

    def neural_network (self, name) :
        return self.session._sqlalchemy_session.query(NeuralNetwork).filter_by(name=str(name)).first()

    def _ids_for_seqs (self, seqs) :
        for seq in seqs :
            try :
                yield seq._id
            except AttributeError :
                if seq is None :
                    yield seq

    def nodes_for_seqs (self, neural_network, seqs, input=None) :

        ids = set(self._ids_for_seqs(seqs))

        if len(ids) :
            io_node_query = self.session._sqlalchemy_session.query(IONode)

            input_filter = () if input is None else (IONode.is_input == input,)

            return io_node_query.filter(IONode.neural_network == neural_network,
                                        or_(IONode._seq_id.in_(ids), IONode._seq_id == None),
                                        *input_filter).all()
        else :
            return []

    def nodes (self, neural_network) :
        return self.session._sqlalchemy_session.query(Node).filter_by(neural_network=neural_network).all()

    def link (self, neural_network, input_node, output_node) :
        return self.session._sqlalchemy_session.query(Link).filter_by(neural_network=neural_network,
                                                                      input_node=input_node,
                                                                      output_node=output_node).first()

def __test__ (ut) :
    from nlplib.core.model.abstract.access import abstract_test
    from nlplib.core.model.sqlalchemy import Database

    abstract_test(ut, Database)

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

