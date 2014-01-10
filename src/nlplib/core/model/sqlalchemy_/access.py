

from sqlalchemy.sql import or_
from sqlalchemy import func

from nlplib.core.model.abstract import access as abstract
from nlplib.core.model import Document, Seq, Gram, Word, Index, NeuralNetwork
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
        return self.session._sqlalchemy_session.query(NeuralNetwork).filter_by(name=name).first()

def __test__ (ut) :
    from nlplib.core.model.abstract.access import abstract_test
    from nlplib.core.model.sqlalchemy_ import Database

    abstract_test(ut, Database)

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

