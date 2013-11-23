''' This module contains multiple handy functions for working with a concordance. '''


from nlplib.core.process.token import split
from nlplib.core.model import Gram
from nlplib.core import Base

__all__ = ['Window', 'raw', 'gram_tuples', 'grams', 'documents_containing']

class Window (Base) :
    ''' This class acts as a window through which you can view the concordance. '''

    __slots__ = ('before', 'after')

    def __init__ (self, before=0, after=2) :
        ''' before : defines how many words/characters before the concordance you can see
            after  : defines how many words/characters after the concordance you can see '''

        self.before = abs(before)
        self.after  = abs(after)

    def __repr__ (self) :
        return super().__repr__(before=self.before, after=self.after)

    def slice (self, start, end=None) :
        if end is None :
            end = start
        return slice(abs(start - self.before), abs(end + self.after + 1))

def raw (concordance) :
    ''' This yields tuples that contain the raw string (unmodified whitespace and all) which the sequence object that
        had made the concordance, was representing. '''

    for document, seq, index in concordance :
        raw_string = str(document)[index.first_character:index.last_character+1]

        yield (document, raw_string, index)

def gram_tuples (concordance, window=Window(), splitter=split) :
    already_split_documents = {}
    for document, seq, index in concordance :
        split_document = already_split_documents.get(document)
        if split_document is None :
            split_document = tuple(splitter(document))
            already_split_documents[document] = split_document

        yield split_document[window.slice(index.first_token, index.last_token)]

def grams (*args, **kw) :
    for gram_tuple in gram_tuples(*args, **kw) :
        yield Gram(gram_tuple)

def documents_containing (concordance) :
    documents = {}
    for document, seq, index in concordance :
        indexes_for_document = documents.setdefault(document, [])
        indexes_for_document.append((seq, index))

    return documents

def _test_window (ut) :
    some_indexes = list(range(10))

    ut.assert_equal(some_indexes[Window(1, 1).slice(5, 7)], [4, 5, 6, 7, 8] )
    ut.assert_equal(some_indexes[Window(2, 2).slice(5)],    [3, 4, 5, 6, 7] )
    ut.assert_equal(some_indexes[Window(1, 1).slice(5, 6)], [4, 5, 6, 7]    )
    ut.assert_equal(some_indexes[Window(-1, 1).slice(5)],   [4, 5, 6]       )
    ut.assert_equal(some_indexes[Window(1, 1).slice(5)],    [4, 5, 6]       )
    ut.assert_equal(some_indexes[Window(0, 1).slice(5)],    [5, 6]          )

def __test__ (ut) :
    from nlplib.core.model import Database, Access, Indexer, Document

    _test_window(ut)

    document_strings = ['Python is a widely used general-purpose, high-level programming language.',
                        ('Its design philosophy emphasizes code readability, and its syntax allows programmers to '
                         'express concepts in fewer lines of code than would be possible in languages such as C.'),
                        ('The language provides constructs intended to enable clear programs on both a small and '
                         'large scale.'),
                        ('Stackless Python is  a significant fork of CPython that implements microthreads; it does not '
                         'use the C memory stack, thus allowing massively concurrent programs. PyPy also has a '
                         'stackless version')]

    db = Database()

    # This builds our index for testing.
    with db as session :
        for document_string in document_strings :
            session.add(Document(document_string))

        indexer = Indexer(session)
        for document in Access(session).all_documents() :
            indexer.add(document)

    # Testing
    with db as session :
        access = Access(session)

        concordance = access.concordance('is a')

        grams_for_concordance = grams(concordance, window=Window(before=1, after=2))
        correct_strings       = ['Python is a widely used', 'Python is a significant fork']
        for gram, correct_string in zip(grams_for_concordance, correct_strings) :
            ut.assert_equal(gram, Gram(correct_string))

        # Tests <raw>, note the differences in the white space.
        for stuff, correct_string in zip(raw(concordance), ['is a', 'is  a']) :
            raw_string = stuff[1]

            ut.assert_equal(raw_string, correct_string)

        documents = documents_containing(concordance)
        ut.assert_equal(len(documents.keys()), 2)

        # Testing by glorified "word" counting.
        ut.assert_equal(len(access.concordance('a')),           4 )
        ut.assert_equal(len(access.concordance('of')),          2 )
        ut.assert_equal(len(access.concordance('to')),          2 )
        ut.assert_equal(len(access.concordance('and')),         2 )
        ut.assert_equal(len(access.concordance('is a')),        2 )
        ut.assert_equal(len(access.concordance('significant')), 1 )
        ut.assert_equal(len(access.concordance('fork of')),     1 )

if __name__ == '__main__' :
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())

