''' This module contains multiple handy functions for working with the concordance of a sequence. '''


from nlplib.core.process.token import split
from nlplib.core.model import Gram
from nlplib.core import Base

__all__ = ['Window', 'concordance', 'raw', 'gram_tuples', 'grams', 'documents_containing']

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

def concordance (seq) :
    try :
        seq.indexes
    except AttributeError :
        pass
    else :
        for index in seq.indexes :
            yield (index.document, index, seq)

def raw (seq) :
    ''' This yields tuples that contain the raw string (unmodified whitespace and all) which the sequence object that
        was representing. '''

    for document, index, seq in concordance(seq) :
        raw_string = str(document)[index.first_character:index.last_character+1]

        yield (document, index, raw_string)

def gram_tuples (seq, window=Window(), splitter=split) :
    already_split_documents = {}
    for document, index, seq in concordance(seq) :
        split_document = already_split_documents.get(document)
        if split_document is None :
            split_document = tuple(splitter(document))
            already_split_documents[document] = split_document

        yield split_document[window.slice(index.first_token, index.last_token)]

def grams (*args, **kw) :
    for gram_tuple in gram_tuples(*args, **kw) :
        yield Gram(gram_tuple)

def documents_containing (seq) :
    documents = {}
    for document, index, seq in concordance(seq) :
        indexes_for_document = documents.setdefault(document, [])
        indexes_for_document.append((index, seq))

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
    from nlplib.core.model import Database, Document
    from nlplib.core.index import Indexed

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

        indexed = Indexed(session)
        for document in session.access.all_documents() :
            indexed.add(document)

    # Testing
    with db as session :
        is_a = session.access.gram('is a')

        grams_for_concordance = grams(is_a, window=Window(before=1, after=2))
        correct_strings       = ['Python is a widely used', 'Python is a significant fork']
        for gram, correct_string in zip(grams_for_concordance, correct_strings) :
            ut.assert_equal(gram, Gram(correct_string))

        # Tests <raw>, note the differences in the white space.
        for stuff, correct_string in zip(raw(is_a), ['is a', 'is  a']) :
            raw_string = stuff[2]

            ut.assert_equal(raw_string, correct_string)

        documents = documents_containing(is_a)
        ut.assert_equal(len(documents.keys()), 2)

        # Testing by glorified "word" counting.
        def test_count (access, string, count) :
            ut.assert_equal(len(list(concordance(access(string)))), count)

        for string, count in [('a', 4), ('of', 2), ('to', 2), ('and', 2), ('significant', 1), ('foobar', 0)] :
            test_count(session.access.word, string, count)

        for string, count in [('is a', 2), ('fork of', 1), ('foo bar', 0)] :
            test_count(session.access.gram, string, count)

if __name__ == '__main__' :
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())

