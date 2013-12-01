

import random

from nlplib.core.process.token import re_tokenize
from nlplib.core.model import SessionDependent, Word, Gram, Index
from nlplib.general.math import hyperbolic
from nlplib.general.iter import windowed

__all__ = ['AddIndexes', 'RemoveIndexes', 'Indexer', 'abstract_test']

class _EditIndexes (SessionDependent) :
    ''' A base class for classes which edit a document's indexes. '''

    def __init__ (self, session, document) :
        super().__init__(session)

        self.document = document

    def __call__ (self) :
        ''' This modifies the indexes for the document (add, remove, update). '''

        raise NotImplementedError

class AddIndexes (_EditIndexes) :
    # Allow for word lemmatization

    stop_seqs = set()

    def __init__ (self, session, document, max_gram=5) :
        super().__init__(session, document)
        self.max_gram = max_gram

    def merge_with_seqs_in_db (self, seqs_with_tokens_from_document) :
        seqs_from_document = (seq for seq, list_of_tokens in seqs_with_tokens_from_document)

        seqs_already_in_db = {str(seq) : seq for seq in self.session.access.matching(seqs_from_document)}

        for seq_from_document, list_of_tokens in seqs_with_tokens_from_document :
            try :
                # The sequence was already in the database, so the sequence object from the database is used.
                seq = seqs_already_in_db[str(seq_from_document)]
            except KeyError :
                # The sequence wasn't in the database, so it's added to the database.
                seq = self.session.add(seq_from_document)
            else :
                seq.prevalence += seq_from_document.prevalence

            yield (seq, list_of_tokens)

    def is_stop_seq (self, seq) :
        return seq in self.stop_seqs

    def tokenize (self, string) :
        return re_tokenize(string)

    def chance (self, prevalence) :
        # z==10, if prevalence < 10 : chance of keeping == 100%
        return random.random() > hyperbolic(y=prevalence, z=10, base=2)

    # give a better name
    def _accumulate (self, seqs, seq, tokens) :
        seq, list_of_tokens = seqs.setdefault(str(seq), (seq, []))
        seq.prevalence += 1
        list_of_tokens.append(tokens)

    # give a better name
    def _gramify (self, tokens, min_size=1) :
        for i in range(min_size, len(tokens) + 1) :
            yield tokens[:i]

    def gram (self, tokens) :
        gram_tuple = tuple(str(token) for token in tokens)
        if not self.is_stop_seq(gram_tuple) :
            return Gram(gram_tuple, prevalence=0)

    def word (self, tokens) :
        word_string = str(tokens[0])
        if not self.is_stop_seq(word_string) :
            return Word(word_string, prevalence=0)

    def seq (self, tokens) :
        if len(tokens) == 1 :
            return self.word(tokens)
        else :
            return self.gram(tokens)

    def seqs_with_tokens (self, document, max_gram) :
        seqs_with_tokens_from_document = {}
        for window in windowed(self.tokenize(document), max_gram) :
            for tokens in self._gramify(window) :
                try :
                    self._accumulate(seqs_with_tokens_from_document, self.seq(tokens), tokens)
                except AttributeError :
                    # <self.seq(tokens)> likely returned None, which means we hit a stop sequence (stop word).
                    pass

        return list(seqs_with_tokens_from_document.values())

    def index (self, *args, **kw) :
        return Index(*args, **kw)

    def make_indexes (self, document, not_yet_indexed) :
        for seq, list_of_tokens in not_yet_indexed :
            for tokens in list_of_tokens :
                first_token = tokens[0]
                last_token  = tokens[-1]

                yield self.index(document,
                                 seq,
                                 first_token.index,
                                 last_token.index,
                                 first_token.first_character_index,
                                 last_token.last_character_index)

class RemoveIndexes (_EditIndexes) :
    def __call__ (self) :
        for index, seq in self.session.access.indexes(self.document) :
            seq.prevalence -= 1

            if seq.prevalence < 1 :
                self.session.remove(seq)

            self.session.remove(index)

class Indexer (SessionDependent) :
    ''' The indexer is used to construct and index of documents within the database. This allows for rapid word and
        gram lookups. '''

    def update (self, document) :
        ''' This updates the indexes for a document. '''

        self.remove(document)
        self.add(document)

    def add (self, document, *args, **kw) :
        ''' This will add an index for each word and gram in a document. Words and grams already in the database will
            have their prevalence scores incremented accordingly.  '''

        raise NotImplementedError

    def remove (self, document) :
        ''' This removes the indexes for a document. This undoes <add>. '''

        raise NotImplementedError

def abstract_test (ut, db_cls) :
    from nlplib.core.model import Document
    from nlplib.core.process.concordance import documents_containing
    from nlplib.core.process.token import re_tokenize

    corpus = [("I'd just like to interject for a moment. What you're referring to as Linux, is in fact, GNU/Linux, or "
               "as I've recently taken to calling it, GNU plus Linux."),
              ('Linux is not an operating system unto itself, but rather another free component of a fully '
               'functioning GNU system made useful by the GNU corelibs, shell utilities and vital system components '
               'comprising a full OS as defined by POSIX.')]

    max_gram = 3

    db = db_cls()

    with db as session :
        for text in corpus :
            session.add(Document(text))

    with db as session :
        for document in session.access.all_documents() :
            add_indexes = AddIndexes(session, document, max_gram=max_gram)

            # This is done in case the default <AddIndexes.tokenize> implementation is changed from <re_tokenize>.
            add_indexes.tokenize = re_tokenize

            add_indexes()

    with db as session :
        ut.assert_equal(max(len(tuple(gram)) for gram in session.access.all_grams()), max_gram)
        ut.assert_equal(len(session.access.all_indexes()), 210)
        from pprint import pprint

        interject = documents_containing(session.access.concordance('interject'))
        ut.assert_equal(len(interject), 1)
        interject_document, indexes = interject.popitem()
        ut.assert_equal(len(indexes), 1)
        interject_word, interject_index = indexes.pop()
        ut.assert_equal((str(interject_document), str(interject_word), int(interject_index)),
                        (corpus[0], 'interject', 5))

        gnu = documents_containing(session.access.concordance('GNU'))
        ut.assert_equal(session.access.word('GNU').prevalence, 4)

        # Sets are used because order is not guaranteed.
        ut.assert_equal({str(document) for document in gnu.keys()},
                        set(corpus))
        ut.assert_equal({(str(word), int(index)) for indexes in gnu.values() for word, index in indexes},
                        {('gnu', 17), ('gnu', 23), ('gnu', 19), ('gnu', 30)})

        ut.assert_true(session.access.word('proprietary') is None)

        ut.assert_equal(len(documents_containing(session.access.concordance('shell utilities')).values()), 1)

    # Test the removal of indexes.
    with db as session :
        for document in access_cls(session).all_documents() :
            RemoveIndexes(session, document)()

    with db as session :
        ut.assert_equal(len(session.access.all_seqs()), 0)
        ut.assert_equal(len(session.access.all_indexes()), 0)
        ut.assert_equal(len(session.access.all_documents()), 2)

