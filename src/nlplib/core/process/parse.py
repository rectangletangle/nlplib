

from nlplib.core.process.token import re_tokenize
from nlplib.core.process import stem
from nlplib.core.model import Word, Gram, Index
from nlplib.general.iter import windowed
from nlplib.core.exc import NLPLibError
from nlplib.core import Base

__all__ = ['DontParse', 'HitStopSeq', 'Parsed']

class DontParse (NLPLibError) :
    pass

class HitStopSeq (DontParse) :
    pass

class Parsed (Base) :
    def __init__ (self, document, stop_seqs=None, max_gram_length=1, yield_grams=True, stem=stem.clean,
                  tokenize=re_tokenize) :

        self.document = document

        if stop_seqs is None :
            self.stop_seqs = set()
        else :
            self.stop_seqs = set(stop_seqs)

        self.max_gram_length = max_gram_length
        self.yield_grams     = yield_grams

        self.stem = stem

        self.tokenize = tokenize

    def __repr__ (self, *args, **kw) :
        # todo : make all repr's take the arguments <*args, **kw>.
        return super().__repr__(self.document, *args, **kw)

    def __iter__ (self) :
        unique_seqs = {}

        for stems, tokens in self._parse() :
            try :
                seq = self._seq(stems)
            except DontParse :
                continue
            else :
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

        for window in windowed(stems_and_tokens, size=max_gram_length) :
            for stems_and_tokens in self._sub_grams(window) :
                stems, tokens = zip(*stems_and_tokens)

                try :
                    self._check_if_is_stop_seq(stems)
                except HitStopSeq :
                    break
                else :
                    yield (stems, tokens)

    def _check_if_is_stop_seq (self, gram_tuple) :
        if len(gram_tuple) == 1 :
            string_or_gram_tuple = gram_tuple[0]
        else :
            string_or_gram_tuple = gram_tuple

        if string_or_gram_tuple in self.stop_seqs :
            raise HitStopSeq

    def _gram (self, gram_tuple) :
        if self.yield_grams :
            return Gram(gram_tuple, count=0)
        else :
            raise DontParse

    def _word (self, word_string) :
        return Word(word_string, count=0)

    def _seq (self, strings) :
        if len(strings) == 1 :
            return self._word(strings[0])
        else :
            return self._gram(strings)

    def _add_index (self, seq, tokens) :
        first_token, last_token = (tokens[0], tokens[-1])

        index = Index(self.document,
                      first_token.index,
                      last_token.index,
                      first_token.first_character_index,
                      last_token.last_character_index)

        seq.count += 1
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
        parsed = Parsed(session.access.all_documents()[0], max_gram_length=max_gram_length)
        unique_seqs = set(parsed)

        words = [seq for seq in unique_seqs if isinstance(seq, Word)]
        grams = [seq for seq in unique_seqs if isinstance(seq, Gram)]

        ut.assert_equal({str(word) for word in words},
                        {stem.clean(token) for token in re_tokenize(text)})

        ut.assert_equal(min(len(gram) for gram in grams), 2)
        ut.assert_equal(max(len(gram) for gram in grams), 7)
        ut.assert_equal(len(words), 26)
        ut.assert_equal(len(grams), 177)

    # The parser shouldn't have any database related side-effects.
    with db as session :
        ut.assert_equal(len(session.access.all_seqs()), 0)
        ut.assert_equal(len(session.access.all_indexes()), 0)
        ut.assert_equal(len(session.access.all_documents()), 1)

    def parsed_string (string, *args, **kw) :
        return ' '.join(str(seq) for seq in Parsed(string, *args, **kw))

    string = 'And The cat ate the  foOd, and'

    for seq, correct_index_count in zip(Parsed(string), [1, 1, 1, 1, 2, 1, 2]) :
        ut.assert_true(seq.count == len(seq.indexes) == correct_index_count)

    ut.assert_equal(parsed_string(string), 'and the cat ate the food and')

    # Testing with stop sequences
    ut.assert_equal(parsed_string(string, stop_seqs={'the', 'ate'}), 'and cat food and')

    # No actual stemming is done if the <str> function is used in place of a stemmer.
    ut.assert_equal(parsed_string(string, stem=str, stop_seqs={'The', 'ate'}), 'And cat the foOd and')

    string = 'the and the and if the and the the and platypus the'
    ut.assert_equal(parsed_string(string, max_gram_length=2, yield_grams=False,
                                  stop_seqs={('and', 'the'), 'platypus'}),
                    'the the and if the the the and the')

if __name__ == '__main__' :
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())

