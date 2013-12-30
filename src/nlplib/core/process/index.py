

from nlplib.core.process.parse import Parsed
from nlplib.core.model import SessionDependent

__all__ = ['Indexed']

class _AddIndexes (SessionDependent) :
    def __init__ (self, session, document, parsed) :
        super().__init__(session)
        self.document = document
        self.parsed = parsed

    def __call__ (self) :
        seqs_from_document = set(self.parsed)

        seqs = self._merge_with_seqs_in_db(seqs_from_document)

        self.document.seqs.extend(seqs)

    def _merge_with_seqs_in_db (self, seqs_from_document) :
        strings_from_document = (str(seq) for seq in seqs_from_document)

        seqs_already_in_db = {(seq.__class__, str(seq)) : seq
                              for seq in self.session.access.matching(strings_from_document)}

        for seq_from_document in seqs_from_document :
            try :
                # The sequence was already in the database, so the sequence object from the database is used.
                seq = seqs_already_in_db[(seq_from_document.__class__, str(seq_from_document))]
            except KeyError :
                # The sequence wasn't in the database, so it's added to the database.
                seq = seq_from_document
            else :
                seq.indexes.extend(seq_from_document.indexes)

            yield seq

class Indexed (SessionDependent) :
    ''' This is used to construct a textual index of documents within the database. This allows for rapid word and
        n-gram (groups of words) lookups. '''

    def _documents (self) :
        return {index.document for index in self.session.access.all_indexes()}

    def __contains__ (self, document) :
        return len(document) and len(self.session.access.indexes(document))

    def __iter__ (self) :
        return iter(self._documents())

    def __len__ (self) :
        return len(self._documents())

    def update (self, document) :
        ''' This updates the indexes for a document. '''

        self.remove(document)
        self.add(document)

    def add (self, document, *args, max_gram_length=5, parser=Parsed, **kw) :
        ''' This will add an index for each word and gram in a document. '''

        _AddIndexes(self.session, document, parser(document, *args, max_gram_length=max_gram_length, **kw))()

        return document

    def remove (self, document) :
        ''' This removes the indexes for a document from the database; this undoes <Indexed.add>.

            Note : This does not remove the document from the database. To remove both the document and its indexes
            simply call <session.remove(document)>. '''

        for object in document.__associated__(self.session) :
            self.session.remove(object)

    def clear (self) :
        for document in self._documents() :
            self.remove(document)

def __test__ (ut) :
    from nlplib.core.model import Document, Database, Word
    from nlplib.core.process.concordance import Concordance

    corpus = [("I'd just like to interject for a moment. What you're referring to as Linux, is in fact, GNU/Linux, or "
               "as I've recently taken to calling it, GNU plus Linux."),

              ('Linux is not an operating system unto itself, but rather another free component of a fully '
               'functioning GNU system made useful by the GNU corelibs, shell utilities and vital system components '
               'comprising a full OS as defined by POSIX.'),

               "This won't be indexed!"]

    db = Database()

    max_gram_length = 3

    def sorted_all_documents (session) :
        return sorted(session.access.all_documents(), key=str)

    with db as session :
        session.add_many(Document(text) for text in corpus)

    with db as session :
        indexed = Indexed(session)
        for document in sorted_all_documents(session)[:-1] :
            indexed.add(document, max_gram_length=max_gram_length)

    with db as session :
        first_document, second_document, third_document = sorted_all_documents(session)
        indexed = Indexed(session)

        ut.assert_true(first_document in indexed)
        ut.assert_true(second_document in indexed)
        ut.assert_true(third_document not in indexed)

        ut.assert_equal(len(indexed), 2)
        ut.assert_equal(set(indexed), {first_document, second_document})

    with db as session :
        ut.assert_equal(max(len(tuple(gram)) for gram in session.access.all_grams()), max_gram_length)
        ut.assert_equal(len(list(session.access.all_indexes())), 210)

        interject_documents = Concordance(session.access.word('interject')).documents()
        ut.assert_equal(len(interject_documents), 1)
        interject_document, indexes = interject_documents.popitem()
        ut.assert_equal(len(indexes), 1)
        interject_index, interject_word = indexes.pop()
        ut.assert_equal((str(interject_document), str(interject_word), int(interject_index)),
                        (corpus[0], 'interject', 5))

        gnu_documents = Concordance(session.access.word('gnu')).documents()
        ut.assert_equal(session.access.word('gnu').count, 4)

        # Sets are used because order is not guaranteed.
        ut.assert_equal({str(document) for document in gnu_documents.keys()},
                        set(sorted(corpus)[:-1]))

        ut.assert_equal({(str(word), int(index)) for indexes in gnu_documents.values() for index, word in indexes},
                        {('gnu', 17), ('gnu', 23), ('gnu', 19), ('gnu', 30)})

        ut.assert_true(session.access.word('proprietary') is None)

        ut.assert_equal(len(Concordance(session.access.gram('shell utilities')).documents().values()), 1)
        ut.assert_equal(len(Concordance(session.access.word('a')).documents().values()), 2)

    # Test the removal of indexes.
    with db as session :
        first_document = sorted_all_documents(session)[0]
        Indexed(session).remove(first_document)

        for seq in session.access.all_seqs() :
            ut.assert_true(not any(index.document is first_document for index in seq.indexes))

    with db as session :
        first_document = sorted_all_documents(session)[0]
        ut.assert_true(first_document not in indexed)
        session.remove(first_document)

    with db as session :
        second_document = sorted_all_documents(session)[0]
        ut.assert_equal(sorted(second_document.seqs), sorted(session.access.all_seqs()))
        ut.assert_true(all(index.document is second_document for index in session.access.all_indexes()))
        Indexed(session).remove(second_document)

    with db as session :
        ut.assert_equal(len(list(session.access.all_seqs())), 0)
        ut.assert_equal(len(list(session.access.all_indexes())), 0)
        ut.assert_equal(len(list(session.access.all_documents())), 2)

        second_document, third_document = sorted_all_documents(session)

        ut.assert_true(second_document not in indexed)
        ut.assert_true(third_document not in indexed)

    db = Database()

    with db as session :
        session.add(Word('a'))
        ut.assert_equal(list(session.access.all_words())[0].count, 0)

    with db as session :
        Indexed(session).add(Document('a a b a'))
        ut.assert_equal(list(session.access.all_words())[0].count, 3)

    with db as session :
        ut.assert_equal(list(session.access.all_words())[0].count, 3)
        session.remove(list(session.access.all_documents())[0])
        ut.assert_equal(list(session.access.all_words()), [])

    with db as session :
        ut.assert_equal(list(session.access.all_words()), [])

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

