

import re

from nlplib.core.base import Base

__all__ = ['Token', 're_tokenized', 'split_tokenized', 'nltk_tokenized', 'split', 'halve', 'map_over_indexes']

class Token (Base) :
    __slots__ = ('string', 'index', 'first_character_index', 'last_character_index')

    def __init__ (self, string, index, first_character_index, last_character_index) :
        self.string = string
        self.index  = index
        self.first_character_index = first_character_index
        self.last_character_index  = last_character_index

    def __repr__ (self, *args, **kw) :
        return super().__repr__(self.string, *args, **kw)

    def __str__ (self) :
        return self.string

    def __len__ (self) :
        return len(self.string)

    def __int__ (self) :
        return self.index

    def slice (self) :
        return slice(self.first_character_index, self.last_character_index + 1)

def re_tokenized (string, pattern=r'\w+(\.?\w+)*') :
    ''' A tokenizer that uses a regular expressions pattern. '''

    for token_index, match in enumerate(re.finditer(str(pattern), str(string))) :
        yield Token(match.group(0), token_index, match.start(), match.end() - 1)

def _tokenized (string, string_tokenize) :
    index_in_string = string.index

    first_index = 0
    for token_index, token_string in enumerate(string_tokenize(string)) :

        first_index = index_in_string(token_string, first_index)
        last_index = first_index + len(token_string) - 1

        yield Token(token_string, token_index, first_index, last_index)

        first_index = last_index

def split_tokenized (string) :
    return _tokenized(string, lambda string : string.split())

try :
    import nltk
except ImportError :
    def nltk_tokenized (*args, **kw) :
        raise ImportError('The function <nltk_tokenized> requires NLTK to be installed.')
else :
    def nltk_tokenized (string) :
        ''' A tokenizer built on top of NLTK's <word_tokenize> function. '''

        return _tokenized(string, nltk.word_tokenize)

def split (string, tokenize=re_tokenized) :
    for token in tokenize(string) :
        yield str(token)

def halve (seq, index) :
    return (seq[:index], seq[index:])

def map_over_indexes (function, seq) :
    for index in range(len(seq)+1) :
        yield function(seq, index)

def __test__ (ut) :
    text = "He was carefully disguised but captured don't quickly by! police."

    tokenized_text = list(re_tokenized(text, pattern=r'\w+(\.?\w+)*'))

    ut.assert_equal([str(token) for token in tokenized_text],
                    ['He', 'was', 'carefully', 'disguised', 'but', 'captured', 'don', 't', 'quickly', 'by', 'police'])

    for tokenized in [tokenized_text, list(split_tokenized(text)), list(nltk_tokenized(text))] :
        for token in tokenized :
            ut.assert_equal(text[token.slice()], str(token))
            ut.assert_equal(str(tokenized[token.index]), str(token))

    ut.assert_equal(list(map_over_indexes(halve, 'hello')),
                    [('', 'hello'), ('h', 'ello'), ('he', 'llo'), ('hel', 'lo'), ('hell', 'o'), ('hello', '')])

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

