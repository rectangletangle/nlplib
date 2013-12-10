

from functools import total_ordering

from nlplib.core.model.base import Model
from nlplib.general import pretty_truncate, literal_representation

class Document (Model) :
    ''' A class for textual documents. '''

    def __init__ (self, string, seqs=(), word_count=None, title=None, url=None, created_on=None) :
        self.string = string
        self.seqs   = list(seqs)
        self.length = len(self.string)

        self.word_count = word_count
        self.title      = title
        self.url        = url
        self.created_on = created_on

    def __repr__ (self) :
        return super().__repr__(pretty_truncate(self.string.replace('\n', ' '), 42))

    def __str__ (self) :
        return self.string

    def __getitem__ (self, index) :
        return self.string[index]

    def __len__ (self) :
        return self.length

@total_ordering
class Seq (Model) :
    ''' This acts as a sequence of characters, similar to a string. The word and gram classes are built on top of
        this. '''

    def __init__ (self, string, count=None, indexes=()) :
        self.string  = string
        self.count   = count
        self.indexes = list(indexes)

    def __repr__ (self) :
        return literal_representation(self, self.string) # Sequence objects can be represented as literal Python.

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
            return self.string == other.string
        except AttributeError :
            return False

    def __lt__ (self, other) :
        try :
            return self.string < other.string
        except AttributeError :
            return True

    def __hash__ (self) :
        return hash(self.string)

class Gram (Seq) :
    ''' This class is used for representing n-grams of word strings. '''

    def __init__ (self, gram_tuple_or_string, *args, **kw) :

        if isinstance(gram_tuple_or_string, str) :
            string = ' '.join(gram_tuple_or_string.split()) # This standardizes the whitespace.
        else :
            string = ' '.join(str(token) for token in gram_tuple_or_string)

        super().__init__(string, *args, **kw)

        self._make_seqs()

    def _make_seqs (self) :
        self.seqs = tuple(self.string.split())

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

class Word (Seq) :
    ''' This class is used for representing words. Currently homographic words are not supported. '''

    pass

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

    def __repr__ (self) :
        return super().__repr__(self.first_token, self.document)

