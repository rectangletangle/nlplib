

import nlplib.core.model

from nlplib.core.process.token import split
from nlplib.core.base import Base

__all__ = ['Window', 'Concordance']

class Window (Base) :
    ''' This class acts as a window through which you can view the concordance. '''

    __slots__ = ('before', 'after')

    def __init__ (self, before=None, after=None) :
        ''' before : how many words/characters before the concordance you can see
            after  : how many words/characters after the concordance you can see '''

        self.before = before if before is not None else 0
        self.after  = after if after is not None else 2

    def __repr__ (self, *args, **kw) :
        return super().__repr__(before=self.before, after=self.after, *args, **kw)

    def slice (self, start, end=None) :
        if end is None :
            end = start
        return slice(abs(start - abs(self.before)), abs(end + abs(self.after) + 1))

class Concordance (Base) :
    ''' This class contains multiple handy methods for working with the concordance of a sequence. '''

    def __init__ (self, seq) :
        self.seq = seq

    def __iter__ (self) :
        try :
            self.seq.indexes
        except AttributeError :
            pass
        else :
            for index in self.seq.indexes :
                yield (index.document, index, self.seq)

    def __len__ (self) :
        try :
            return len(self.seq.indexes)
        except AttributeError :
            return 0

    def __repr__ (self, *args, **kw) :
        return super().__repr__(self.seq, *args, **kw)

    def raw (self) :
        ''' This yields tuples that contain the raw string (unmodified whitespace and all) which the sequence object
            was representing. '''

        for document, index, seq in self :
            raw_string = str(document)[index.first_character:index.last_character+1]
            yield (document, index, raw_string)

    def gram_tuples (self, before=None, after=None, splitter=split) :

        window = Window(before=before, after=after)

        already_split_documents = {}
        for document, index, seq in self :
            try :
                split_document = already_split_documents[document]
            except KeyError :
                split_document = tuple(splitter(document))
                already_split_documents[document] = split_document

            yield split_document[window.slice(index.first_token, index.last_token)]

    def grams (self, *args, **kw) :
        gram_cls = nlplib.core.model.Gram
        for gram_tuple in self.gram_tuples(*args, **kw) :
            yield gram_cls(gram_tuple)

    def documents (self, documents=None) :
        if documents is None :
            documents = {}

        for document, index, seq in self :
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
    from nlplib.core.process.index import Indexed
    from nlplib.core.model import Database, Document, Word, Gram

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
            indexed.add(document, max_gram_length=5)

    # Testing
    with db as session :
        is_a = session.access.gram('is a')

        concordance_of_is_a = Concordance(is_a)
        ut.assert_equal(len(concordance_of_is_a), 2)

        ut.assert_equal(list(concordance_of_is_a), list(is_a.concordance()))

        grams_for_concordance = concordance_of_is_a.grams(before=1, after=2)
        correct_strings = ['Python is a widely used', 'Python is a significant fork']
        for gram, correct_string in zip(grams_for_concordance, correct_strings) :
            ut.assert_equal(gram, Gram(correct_string))

        # Tests <raw>, note the differences in the white space.
        for stuff, correct_string in zip(concordance_of_is_a.raw(), ['is a', 'is  a']) :
            raw_string = stuff[2]

            ut.assert_equal(raw_string, correct_string)

        ut.assert_equal(len(concordance_of_is_a.documents().keys()), 2)

        session.add(Word('foobar'))
        session.add(Gram('foo bar'))

        # Testing by glorified "word" counting.
        def test_count (access, string, count) :
            ut.assert_equal(len(Concordance(access(string))), count)

        for string, count in [('a', 4), ('of', 2), ('to', 2), ('and', 2), ('significant', 1), ('foobar', 0)] :
            test_count(session.access.word, string, count)

        for string, count in [('is a', 2), ('fork of', 1), ('foo bar', 0)] :
            test_count(session.access.gram, string, count)

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

