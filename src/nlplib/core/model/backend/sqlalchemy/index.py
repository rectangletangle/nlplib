

from nlplib.core.model.backend.sqlalchemy.map import default_mapper
from nlplib.core.model.backend.abstract import index as abstract

__all__ = ['AddIndexes', 'RemoveIndexes', 'Indexer']

class AddIndexes(abstract.AddIndexes) :

    def index (self, document, seq, first_token, last_token, first_character, last_character) :
        return {'document_id'            : document.id,
                'seq_id'                 : seq.id,
                'first_token'            : first_token,
                'last_token'             : last_token,
                'first_character'        : first_character,
                'last_character'         : last_character,
                'tokenization_algorithm' : None}

    def __call__ (self) :
        seqs_with_tokens_from_document = self.seqs_with_tokens(self.document, self.max_gram)
        not_yet_indexed = list(self.merge_with_seqs_in_db(seqs_with_tokens_from_document))

        self.add_indexes_to_db(self.make_indexes(self.document, not_yet_indexed))

    def add_indexes_to_db (self, indexes) :
        self.session._sqlalchemy_session.execute(default_mapper.tables['index'].insert(), list(indexes))

class RemoveIndexes (abstract.RemoveIndexes) :
    # todo : Not necessary, no longer implementation specific.
    pass

class Indexer (abstract.Indexer) :
    def add (self, document, *args, **kw) :
        AddIndexes(self.session, document, *args, **kw)()

    def remove (self, document) :
        RemoveIndexes(self.session, document)()

def __test__ (ut) :
    from nlplib.core.model.backend.abstract.access import abstract_test
    from nlplib.core.model.backend.sqlalchemy import Database

    abstract_test(ut, Database)

if __name__ == '__main__' :
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())

