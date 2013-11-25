

from nlplib.core.model import Document, Seq, Gram, Word, Index
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
        return [(seq, index) for index, seq
                in session.query(Index, Seq).filter(Index.document_id == document.id).join(Seq).all()]

    def matching (self, strings, cls=Seq, chunk_size=200) :
        for chunked_strings in chunked((process(string) for string in strings), chunk_size) :
            for match in self.session._sqlalchemy_session.query(cls).filter(cls.clean.in_(chunked_strings)).all() :
                yield match

def __test__ (ut) :
    from nlplib.core.model.backend.abstract.access import abstract_test
    from nlplib.core.model.backend.sqlalchemy import Database

    abstract_test(ut, Database, Access)

if __name__ == '__main__' :
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())

