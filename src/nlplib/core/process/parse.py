

from nlplib.core.process.token import re_tokenize
from nlplib.core.process import stem
from nlplib.core.model import Word, Gram, Index
from nlplib.general.iter import windowed
from nlplib.core import Base

__all__ = ['Parse']

class Parse (Base) :
    def __init__ (self, document, stop_seqs=None, max_gram_length=5) :
        self.document = document

        if stop_seqs is None :
            self.stop_seqs = set()
        else :
            self.stop_seqs = stop_seqs

        self.max_gram_length = max_gram_length

    def _stem_tokens (self, tokens) :
        for token in tokens :
            yield (self.stem(str(token)), token)

    def _sub_grams (self, tuple_) :
        min_gram_length = 1
        for i in range(min_gram_length, len(tuple_) + 1) :
            yield tuple_[:i]

    def _accumulate_seqs (self, seqs_with_tokens_from_document, seq, tokens) :
        seq, all_tokens_for_seq = seqs_with_tokens_from_document.setdefault(str(seq), (seq, []))
        seq.count += 1
        all_tokens_for_seq.append(tokens)

    def __call__ (self) :
        seqs_with_tokens_from_document = {}

        stems_and_tokens = self._stem_tokens(self.tokenize(self.document))

        for window in windowed(stems_and_tokens, size=self.max_gram_length) :
            for stems_and_tokens in self._sub_grams(window) :
                stems, tokens = zip(*stems_and_tokens)

                try :
                    self._accumulate_seqs(seqs_with_tokens_from_document, self.seq(stems), tokens)
                except AttributeError :
                    # <self.seq(tokens)> likely returned None, which means we hit a stop sequence (stop word).
                    pass

        for seq, all_tokens_for_seq in seqs_with_tokens_from_document.values() :
            seq.indexes.extend(self.indexes(all_tokens_for_seq))
            yield seq

    def is_stop_seq (self, seq) :
        return seq in self.stop_seqs

    def tokenize (self, string) :
        ''' This is a proxy to the specific tokenization algorithm used to split the document into tokens (think
            words). '''

        return re_tokenize(string)

    def stem (self, string) :
        ''' This is a proxy to the specific stemming/lemmatisation algorithm used to map multiple forms of a word,
            e.g., diffrent capitlization, whitespace, or inflections, to a single form. '''

        return stem.clean(string)

    def gram (self, gram_tuple) :
        if not self.is_stop_seq(gram_tuple) :
            return Gram(gram_tuple, count=0)

    def word (self, word_string) :
        if not self.is_stop_seq(word_string) :
            return Word(word_string, count=0)

    def seq (self, strings) :
        if len(strings) == 1 :
            return self.word(strings[0])
        else :
            return self.gram(strings)

    def indexes (self, all_tokens_for_seq) :
        for group_of_tokens in all_tokens_for_seq :
            first_token, last_token = (group_of_tokens[0], group_of_tokens[-1])

            yield Index(self.document,
                        first_token.index,
                        last_token.index,
                        first_token.first_character_index,
                        last_token.last_character_index)

def __test__ (ut) :
    from nlplib.core.model import Document, Database
    from nlplib.core.process.concordance import documents_containing
    from nlplib.core.process.token import re_tokenize

    text = ("I'd just like to interject for a moment. What you're referring to as Linux, is in fact, GNU/Linux, or "
            "as I've recently taken to calling it, GNU plus Linux.")

    max_gram_length = 7

    db = Database()

    with db as session :
        session.add(Document(text))

    with db as session :
        seqs = list(Parse(session.access.all_documents()[0], max_gram_length=max_gram_length)())

        words = [seq for seq in seqs if isinstance(seq, Word)]
        grams = [seq for seq in seqs if isinstance(seq, Gram)]

        ut.assert_equal({str(word) for word in words},
                        {stem.clean(token) for token in re_tokenize(text)})

        ut.assert_equal(min(len(gram) for gram in grams), 2)
        ut.assert_equal(max(len(gram) for gram in grams), 7)
        ut.assert_equal(len(words), 26)
        ut.assert_equal(len(grams), 177)

    with db as session :
        ut.assert_equal(len(session.access.all_seqs()), 0)
        ut.assert_equal(len(session.access.all_indexes()), 0)
        ut.assert_equal(len(session.access.all_documents()), 1)

if __name__ == '__main__' :
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())

