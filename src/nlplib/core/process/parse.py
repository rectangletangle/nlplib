

from nlplib.core.process.token import re_tokenized
from nlplib.core.process import stem
from nlplib.core.model import Word, Gram, Index
from nlplib.general.iterate import windowed
from nlplib.core.base import Base

__all__ = ['Parsed']

class Parsed (Base) :
    def __init__ (self, document, max_gram_length=1, stem=stem.clean, tokenize=re_tokenized) :

        self.document = document

        self.max_gram_length = max_gram_length

        self.stem = stem
        self.tokenize = tokenize

    def __repr__ (self, *args, **kw) :
        return super().__repr__(self.document, *args, **kw)

    def __iter__ (self) :
        unique_seqs = {}

        for stems, tokens in self._parse() :

            seq = self._make_seq(stems)

            # This makes sure that sequences that have the same string value map to a single sequence instance.
            seq = unique_seqs.setdefault(str(seq), seq)

            self._add_index(seq, tokens)

            yield seq

    def _stem_tokens (self, tokens) :
        for token in tokens :
            yield (self.stem(str(token)), token)

    def _sub_grams (self, tuple_) :
        min_gram_length = 1
        for i in range(min_gram_length, len(tuple_) + 1) :
            yield tuple_[:i]

    def _sub_grams (self, tuple_) :
        min_gram_length = 1
        for i in range(len(tuple_), min_gram_length-1, -1) :
            yield tuple_[:i]

    def _parse (self) :
        stems_and_tokens = self._stem_tokens(self.tokenize(self.document))
        max_gram_length = self.max_gram_length

        for window in windowed(stems_and_tokens, size=max_gram_length, trail=True) :
            for stems_and_tokens in self._sub_grams(window) :
                stems, tokens = zip(*stems_and_tokens)

                yield (stems, tokens)

    def _make_seq (self, stems) :
        return Word(stems[0]) if len(stems) == 1 else Gram(stems)

    def _add_index (self, seq, tokens) :
        first_token, last_token = (tokens[0], tokens[-1])

        index = Index(self.document,
                      first_token.index,
                      last_token.index,
                      first_token.first_character_index,
                      last_token.last_character_index,
                      self.tokenize.__name__)

        seq.indexes.append(index)

def __test__ (ut) :
    from nlplib.core.model import Document, Database

    text = ("I'd just like to interject for a moment. What you're referring to as Linux, is in fact, GNU/Linux, or "
            "as I've recently taken to calling it, GNU plus Linux.")

    max_gram_length = 7

    db = Database()

    with db as session :
        session.add(Document(text))

    with db as session :
        parsed = Parsed(list(session.access.all_documents())[0], max_gram_length=max_gram_length, stem=stem.clean,
                             tokenize=re_tokenized)
        unique_seqs = set(parsed)

        words = [seq for seq in unique_seqs if isinstance(seq, Word)]
        grams = [seq for seq in unique_seqs if isinstance(seq, Gram)]

        ut.assert_equal({str(word) for word in words},
                        {stem.clean(token) for token in re_tokenized(text)})

        ut.assert_equal(min(len(gram) for gram in grams), 2)
        ut.assert_equal(max(len(gram) for gram in grams), 7)
        ut.assert_equal(len(words), 26)
        ut.assert_equal(len(grams), 177)

        ut.assert_true(all(index.tokenization_algorithm == 're_tokenized'
                           for seq in parsed for index in seq.indexes))

    # The parser shouldn't have any database related side-effects.
    with db as session :
        ut.assert_equal(len(list(session.access.all_seqs())), 0)
        ut.assert_equal(len(list(session.access.all_indexes())), 0)
        ut.assert_equal(len(list(session.access.all_documents())), 1)

    def parsed_string (string, *args, **kw) :
        return ' '.join(str(seq) for seq in Parsed(string, *args, **kw))

    string = 'And The cat ate the  foOd, and'

    for seq, correct_index_count in zip(Parsed(string), [1, 1, 1, 1, 2, 1, 2]) :
        ut.assert_true(seq.count == len(seq.indexes) == correct_index_count)

    ut.assert_equal(parsed_string(string), 'and the cat ate the food and')

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

