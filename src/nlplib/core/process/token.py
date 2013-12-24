

import re

from nlplib.core.base import Base

__all__ = ['Token', 're_tokenize', 'split', 'halve', 'map_over_indexes']

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

def re_tokenize (string, pattern=r'\w+(\.?\w+)*') :
    for token_index, match in enumerate(re.finditer(str(pattern), str(string))) :
        yield Token(match.group(0), token_index, match.start(), match.end() - 1)

def split (string, tokenize=re_tokenize) :
    for token in tokenize(string) :
        yield str(token)

def halve (seq, index) :
    return (seq[:index], seq[index:])

def map_over_indexes (function, seq) :
    for index in range(len(seq)+1) :
        yield function(seq, index)

def __test__ (ut) :
    text = "He was carefully disguised but captured don't quickly by! police."

    tokenized_text = list(re_tokenize(text, pattern=r'\w+(\.?\w+)*'))

    ut.assert_equal([str(token) for token in tokenized_text],
                    ['He', 'was', 'carefully', 'disguised', 'but', 'captured', 'don', 't', 'quickly', 'by', 'police'])

    for token in tokenized_text :
        ut.assert_equal(text[token.slice()], str(token))
        ut.assert_equal(str(tokenized_text[token.index]), str(token))

    ut.assert_equal(list(map_over_indexes(halve, 'hello')),
                    [('', 'hello'), ('h', 'ello'), ('he', 'llo'), ('hel', 'lo'), ('hell', 'o'), ('hello', '')])

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

