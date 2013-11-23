

from nlplib.core.process.token import split
from nlplib.core.model import SessionDependent

__all__ = ['Access', 'abstract_test']

class Access (SessionDependent) :
    ''' This class contains methods which act as convenient abstractions of common database queries; this provides the
        primary way to access the objects stored within a database. '''

    def words (self, string, splitter=split) :
        ''' This returns the word objects corresponding to the word substrings within a string. If no word object is
            found for a particular substring, <None> is used. '''

        return [self.word(word_string) for word_string in splitter(string)]

    def vocabulary (self) :
        return self.all_words()

    def all_documents (self) :
        ''' This returns all of the document objects in the database. '''

        raise NotImplementedError

    def all_seqs (self) :

        raise NotImplementedError

    def all_grams (self) :
        ''' This returns all of the gram objects in the database. '''

        raise NotImplementedError

    def all_words (self) :
        ''' This returns all of the word objects in the database. '''

        raise NotImplementedError

    def all_indexes (self) :

        raise NotImplementedError

    def specific (self, cls, id) :
        ''' This returns a specific object by id. '''

        raise NotImplementedError

    def most_prevalent (self, cls=None, top=1000) :
        ''' This returns most prevalent (commonly encountered) objects. '''

        raise NotImplementedError

    def gram (self, gram_string_or_tuple) :
        ''' This returns the gram object corresponding to a string or tuple, or <None>.
            gram_string = 'the cat ate'
            gram_tuple  = ('the', 'cat', 'ate') '''

        raise NotImplementedError

    def word (self, word_string) :
        ''' This returns the word object corresponding to a string, or <None>. '''

        raise NotImplementedError

    def concordance (self, string) :

        raise NotImplementedError

    def indexes (self, document) :
        ''' This returns all of the indexes referencing the document. '''

        raise NotImplementedError

    def matching (self, strings, cls=None, chunk_size=200) :
        ''' This returns sequences (grams and words) that match the given list of strings.

            Note : This method is typically implemented using the SQL <IN> operator. Some database systems have
            stipulations regarding the maximum size of the set used for membership testing. The optional <chunk_size>
            argument allows the set to be broken up into multiple smaller sets (chunks), with a length corresponding to
            <chunk_size>, so that the set may fall under this limit. '''

        raise NotImplementedError

def abstract_test (ut, db_cls, access_cls) :

    from nlplib.core.model import Seq, Gram, Word

    chars = 'abc'

    db = db_cls()

    with db as session :
        for prevalence, char in enumerate(chars, 1) :
            for cls in (Seq, Gram, Word) :
                session.add(cls(char, prevalence=prevalence))

    def mock (classes, chars) :
        return sorted(cls(char) for char in chars for cls in classes)

    with db as session :
        access = access_cls(session)

        ut.assert_equal(sorted(access.all_seqs()),
                        mock((Seq, Gram, Word), chars))
        ut.assert_equal(sorted(access.all_grams()),
                        mock((Gram,), chars))
        ut.assert_equal(sorted(access.all_words()),
                        mock((Word,), chars))
        ut.assert_equal(sorted(access.vocabulary()),
                        sorted(access.all_words()))

        ut.assert_equal(sorted(access.most_prevalent(cls=Seq, top=3)),
                        mock((Seq, Gram, Word), 'c'))

        ut.assert_equal(sorted(access.most_prevalent(cls=Word, top=2)),
                        mock((Word,), 'bc'))

        ut.assert_equal(access.word('a'), Word('a'))
        ut.assert_equal(access.word('z'), None)

        ut.assert_equal(access.words('b a c'), [Word('b'), Word('a'), Word('c')])
        ut.assert_equal(access.words('b z c'), [Word('b'), None, Word('c')])
        ut.assert_true(access.words('b a c') != [Word('a'), Word('b'), Word('c')])
        ut.assert_equal(access.words(''), [])

        ut.assert_equal(sorted(access.matching(['a', 'b'])), mock((Seq, Gram, Word), 'ab'))
        ut.assert_equal(sorted(access.matching(['a', 'b'], Word)), mock((Word,), 'ab'))
        ut.assert_equal(sorted(access.matching([])), [])

