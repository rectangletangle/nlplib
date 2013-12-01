
from nlplib.core.model import Document, Seq, Gram, Word, Index, NeuralNetwork, Node, IONode, Link
from nlplib.core.model.backend.abstract import access as abstract
from nlplib.core.process import process
from nlplib.general.iter import chunked

__all__ = ['Access']

class Access (abstract.Access) :
    def _all (self, cls) :
        return self.session._sqlalchemy_session.query(cls).all()

    def all_documents (self) :
        return self._all(Document)

    def all_seqs (self) :
        return self._all(Seq)

    def all_grams (self) :
        return self._all(Gram)

    def all_words (self) :
        return self._all(Word)

    def all_indexes (self) :
        return self._all(Index)

    def all_neural_networks (self) :
        return self._all(NeuralNetwork)

    def specific (self, cls, id) :
        return self.session._sqlalchemy_session.query(cls).get(id)

    def most_prevalent (self, cls=Seq, top=1000) :
        return self.session._sqlalchemy_session.query(cls).order_by(cls.prevalence.desc()).slice(0, top).all()

    def gram (self, gram_string_or_tuple) :
        session = self.session._sqlalchemy_session
        return session.query(Gram).filter_by(clean=str(Gram(gram_string_or_tuple))).first()

    def word (self, word_string) :
        return self.session._sqlalchemy_session.query(Word).filter_by(clean=process(word_string)).first()

    def concordance (self, string) :
        session = self.session._sqlalchemy_session
        concordance = session.query(Document, Seq, Index).join(Index).join(Seq).filter_by(clean=process(string)).all()

        return [(document, index, seq) for document, seq, index in concordance]

    def indexes (self, document) :
        session = self.session._sqlalchemy_session
        return session.query(Index, Seq).filter(Index.document_id == document.id).join(Seq).all()

    def matching (self, strings, cls=Seq, chunk_size=200) :
        for chunked_strings in chunked((process(string) for string in strings), chunk_size) :
            for match in self.session._sqlalchemy_session.query(cls).filter(cls.clean.in_(chunked_strings)).all() :
                yield match

    def neural_network (self, name) :
        return self.session._sqlalchemy_session.query(NeuralNetwork).filter_by(name=str(name)).first()

    def nodes_in_layer (self, neural_network, layer_index) :
        session = self.session._sqlalchemy_session
        return session.query(Node).filter_by(neural_network_id=neural_network.id, layer_index=layer_index).all()

    def nodes_for_seqs (self, neural_network, seqs, layer_index=None) :
        # todo : This probably could be made more efficient using the SQL <in> operator.

        neural_network_id = neural_network.id

        query_nodes = self.session._sqlalchemy_session.query(IONode).filter_by

        if layer_index is None :
            for seq in seqs :
                for node in query_nodes(neural_network_id=neural_network_id, seq_id=seq.id).all() :
                    yield node
        else :
            for seq in seqs :
                for node in query_nodes(neural_network_id=neural_network_id, layer_index=layer_index,
                                        seq_id=seq.id).all() :
                    yield node

    def link (self, neural_network, input_node, output_node) :
        return self.session._sqlalchemy_session.query(Link).filter_by(neural_network_id=neural_network.id,
                                                                      input_node_id=input_node.id,
                                                                      output_node_id=output_node.id).first()

def __test__ (ut) :
    from nlplib.core.model.backend.abstract.access import abstract_test
    from nlplib.core.model.backend.sqlalchemy import Database

    abstract_test(ut, Database, Access)

if __name__ == '__main__' :
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())

