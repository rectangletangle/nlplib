

from functools import total_ordering

from nlplib.core.model.base import Model
from nlplib.general.represent import pretty_truncate, represented_literally
from nlplib.general import composite
from nlplib.core.base import Base

def _SeqsMixin (Base) : # todo :
    def seqs_only (self) :
        for seq in self.seqs :
            if seq._is_seq :
                yield seq

    def words (self) :
        for word in self.seqs :
            if word._is_word :
                yield word

    def grams (self) :
        for gram in self.seqs :
            if gram._is_gram :
                yield gram

class Document (Model) :
    ''' A class for textual documents. '''

    # todo : make take parsed config, similar to nn structure config

    def __init__ (self, string, word_count=None, title=None, url=None, created_on=None) :
        self.string = string

        self.word_count = word_count
        self.title      = title
        self.url        = url
        self.created_on = created_on

        self.seqs = [] # todo : make composite property from indexed

    def __repr__ (self, *args, **kw) :
        return super().__repr__(pretty_truncate(self.string.replace('\n', ' '), 35), *args, **kw)

    def __str__ (self) :
        return self.string

    def __getitem__ (self, index) :
        return self.string[index]

    def __contains__ (self, seq) :
        return seq in self.seqs

    def __len__ (self) :
        return len(self.string)

    def _associated (self, session) :
        for index, seq in session.access.indexes(self) :
            seq.indexes.remove(index)
            if seq.count < 1 :
                yield seq
            yield index

@total_ordering
class Seq (Model) :
    ''' This acts as a sequence of characters, similar to a string. The word and gram classes are built on top of
        this. '''

    _is_seq  = True
    _is_gram = False
    _is_word = False

    def __init__ (self, string) :
        self.string = string

        self.indexes = []

    def __repr__ (self, *args, **kw) :
        # Sequence objects can be represented as literal Python.
        return represented_literally(self, self.string, *args, **kw)

    def __str__ (self) :
        return self.string

    def __getitem__ (self, index) :
        return self.string[index]

    def __len__ (self) :
        return len(self.string)

    def __add__ (self, other) :
        if isinstance(other, Gram) :
            return Gram((self,) + tuple(other))
        else :
            return Gram((self, other))

    def __eq__ (self, other) :
        try :
            return self.__class__ is other.__class__ and self.string == str(other)
        except AttributeError :
            return False

    def __lt__ (self, other) :
        if self.__class__ is other.__class__ :
            return self.string < str(other)
        else :
            try :
                cls_index = _seq_cls_order.index(self.__class__)
                other_cls_index = _seq_cls_order.index(other.__class__)
            except ValueError :
                return True
            else :
                return cls_index < other_cls_index

    def __hash__ (self) :
        return hash((self.__class__, self.string))

    @composite(lambda self : (len(self.indexes),))
    def count (self) :
        return len(self.indexes)

    def concordance (self) : # todo :
        raise NotImplementedError

class Word (Seq) :
    ''' This class is used for representing words. Currently homographic words are not supported. '''

    _is_seq  = False
    _is_gram = False
    _is_word = True

class Gram (Seq) :
    ''' This class is used for representing n-grams of word strings. '''

    _is_seq  = False
    _is_gram = True
    _is_word = False

    def __init__ (self, gram_tuple_or_string, *args, **kw) :

        if isinstance(gram_tuple_or_string, str) :
            string = ' '.join(gram_tuple_or_string.split()) # This standardizes the whitespace.
        else :
            string = ' '.join(str(token) for token in gram_tuple_or_string)

        super().__init__(string, *args, **kw)

    def __iter__ (self) :
        return iter(self.seqs)

    def __len__ (self) :
        return len(self.seqs)

    def __getitem__ (self, index) :
        if isinstance(index, slice) :
            return Gram(self.seqs[index])
        else :
            return self.seqs[index]

    def __add__ (self, other) :
        if isinstance(other, Gram) :
            other = tuple(other)
        else :
            other = (other,)

        return Gram(self.seqs + other)

    @composite(lambda self : (self.string,))
    def seqs (self) :
        return tuple(self.string.split())

_seq_cls_order = (Seq, Word, Gram)

class Index (Model) :
    ''' This class is used for indexing sequences (words or n-grams) in a document. '''

    def __init__ (self, document, first_token_index, last_token_index, first_character_index,
                  last_character_index, tokenization_algorithm=None) :

        self.document = document

        ''' Token indexes are the indexes of the sequence, if the document were to be represented as a list of tokens.

            Note : The index value is dependent on the exact tokenization algorithm used to make the index.

            document = ['i', 'may', 'be', 'a', 'fish', '?']
            first_token_index = 4

            assert 'fish' == document[first_token_index] '''

        self.first_token = self._int_or_none(first_token_index)
        self.last_token  = self._int_or_none(last_token_index)

        ''' The character index is the same as the index returned by <string.index(some_substring)>. '''

        self.first_character = self._int_or_none(first_character_index)
        self.last_character  = self._int_or_none(last_character_index)

        self.tokenization_algorithm = tokenization_algorithm

    def _int_or_none (self, value) :
        try :
            return int(value)
        except TypeError :
            return None

    def __int__ (self) :
        return self.first_token

    def __len__ (self) :
        return self.last_token - self.first_token

    def __repr__ (self, *args, **kw) :
        return super().__repr__(self.first_token, self.document, *args, **kw)

