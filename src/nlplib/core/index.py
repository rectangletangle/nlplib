

from nlplib.core.process.parse import Parse
from nlplib.core.model import SessionDependent

__all__ = ['AddIndexes', 'RemoveIndexes', 'Indexed']

class _EditIndexes (SessionDependent) :
    ''' A base class for classes which edit a document's indexes. '''

    def __init__ (self, session, document) :
        super().__init__(session)

        self.document = document

    def __call__ (self) :
        ''' This modifies the indexes for the document (add, remove, update). '''

        raise NotImplementedError

class AddIndexes (_EditIndexes) :
    ''' This will add an index for each word and gram in a document. Words and grams already in the database will have
        their count incremented accordingly. '''

    def __init__ (self, session, document, *args, **kw) :
        super().__init__(session, document)
        self.parser = Parse(self.document, *args, **kw)

    def __call__ (self) :
        seqs_from_document = list(self.parser())

        seqs = self._merge_with_seqs_in_db(seqs_from_document)

        self.document.seqs.extend(seqs)

    def _merge_with_seqs_in_db (self, seqs_from_document) :
        strings_from_document = (str(seq) for seq in seqs_from_document)

        seqs_already_in_db = {(seq.type, str(seq)) : seq
                              for seq in self.session.access.matching(strings_from_document)}

        for seq_from_document in seqs_from_document :
            try :
                # The sequence was already in the database, so the sequence object from the database is used.
                seq = seqs_already_in_db[(seq_from_document.type, str(seq_from_document))]
            except KeyError :
                # The sequence wasn't in the database, so it's added to the database.
                seq = seq_from_document
            else :
                seq.count += seq_from_document.count
                seq.indexes.extend(seq_from_document.indexes)

            yield seq

class RemoveIndexes (_EditIndexes) :
    ''' This removes the indexes for a document; this undoes <AddIndexes>. '''

    def __call__ (self) :
        seqs = list(self.document.seqs)
        self.document.seqs.clear()

        for seq in seqs :
            for index in list(seq.indexes) :
                if index.document is self.document :
                    seq.indexes.remove(index)
                    self.session.remove(index)
                    seq.count -= 1

            if seq.count < 1 :
                self.session.remove(seq)

class Indexed (SessionDependent) :
    ''' This is used to construct a textual index of documents within the database. This allows for rapid word and
        n-gram (groups of words) lookups. '''

    def update (self, document) :
        ''' This updates the indexes for a document. '''

        self.remove(document)
        self.add(document)

    def add (self, document, *args, **kw) :
        AddIndexes(self.session, document, *args, **kw)()

    def remove (self, document) :
        RemoveIndexes(self.session, document)()

def __test__ (ut) :
    from nlplib.core.model import Document, Database
    from nlplib.core.process.concordance import documents_containing
    from nlplib.core.process.token import re_tokenize

    corpus = [("I'd just like to interject for a moment. What you're referring to as Linux, is in fact, GNU/Linux, or "
               "as I've recently taken to calling it, GNU plus Linux."),
              ('Linux is not an operating system unto itself, but rather another free component of a fully '
               'functioning GNU system made useful by the GNU corelibs, shell utilities and vital system components '
               'comprising a full OS as defined by POSIX.')]

    db = Database()

    max_gram_length = 3

    with db as session :
        session.add_many(Document(text) for text in corpus)

    with db as session :
        for document in session.access.all_documents() :
            AddIndexes(session, document, max_gram_length=max_gram_length)()

    with db as session :
        ut.assert_equal(max(len(tuple(gram)) for gram in session.access.all_grams()), max_gram_length)
        ut.assert_equal(len(session.access.all_indexes()), 210)

        interject = documents_containing(session.access.word('interject'))
        ut.assert_equal(len(interject), 1)
        interject_document, indexes = interject.popitem()
        ut.assert_equal(len(indexes), 1)
        interject_index, interject_word = indexes.pop()
        ut.assert_equal((str(interject_document), str(interject_word), int(interject_index)),
                        (corpus[0], 'interject', 5))

        gnu = documents_containing(session.access.word('gnu'))
        ut.assert_equal(session.access.word('gnu').count, 4)

        # Sets are used because order is not guaranteed.
        ut.assert_equal({str(document) for document in gnu.keys()},
                        set(corpus))

        ut.assert_equal({(str(word), int(index)) for indexes in gnu.values() for index, word in indexes},
                        {('gnu', 17), ('gnu', 23), ('gnu', 19), ('gnu', 30)})

        ut.assert_true(session.access.word('proprietary') is None)

        ut.assert_equal(len(documents_containing(session.access.gram('shell utilities')).values()), 1)
        ut.assert_equal(len(documents_containing(session.access.word('a')).values()), 2)

    # Test the removal of indexes.
    with db as session :
        sorted_all_documents = sorted(session.access.all_documents(), key=lambda document : str(document))
        first_document = sorted_all_documents[0]
        RemoveIndexes(session, first_document)()
        session.remove(first_document)

    with db as session :
        other_document = session.access.all_documents()[0]
        ut.assert_equal(sorted(other_document.seqs), sorted(session.access.all_seqs()))
        ut.assert_true(all(index.document is other_document for index in session.access.all_indexes()))
        RemoveIndexes(session, other_document)()

    with db as session :
        ut.assert_equal(len(session.access.all_seqs()), 0)
        ut.assert_equal(len(session.access.all_indexes()), 0)
        ut.assert_equal(len(session.access.all_documents()), 1)

if __name__ == '__main__' :
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())

