

from nlplib.core.process.token import re_tokenize
from nlplib.core.process import stem
from nlplib.core.model import Word, Gram, Index
from nlplib.general.iter import windowed
from nlplib.core import Base


__all__ = ['Parse']

class Parse (Base) :
    def __init__ (self, document) :
        self.document = document

    def _accumulate_seqs (self, seqs_with_tokens_from_document, seq, tokens) :
        seq, list_of_tokens = seqs_with_tokens_from_document.setdefault(str(seq), (seq, []))
        seq.prevalence += 1
        list_of_tokens.append(tokens)

    def _group_tokens_like_a_gram (self, tokens, min_size=1) :
        for i in range(min_size, len(tokens) + 1) :
            yield tokens[:i]

    def __call__ (self) :
        seqs_with_tokens_from_document = {}

        stems_and_tokens = self._stem_tokens(self.tokenize(document))



        for window in windowed(stems_and_tokens, size=self.max_gram_length) :
            for tokens in self._group_tokens_like_a_gram(window) :
                try :
                    self._accumulate_seqs(seqs_with_tokens_from_document, self.seq(tokens), tokens)
                except AttributeError :
                    # <self.seq(tokens)> likely returned None, which means we hit a stop sequence (stop word).
                    pass

        return list(seqs_with_tokens_from_document.values())


    max_gram_length = 5

    stop_seqs = set()

    def is_stop_seq (self, seq) :
        return seq in self.stop_seqs

    def _stem_tokens (self, tokens) :
        for token in tokens :
            yield (self.stem(str(token)), token)

    def tokenize (self, string) :
        ''' This is a proxy to the specific tokenization algorithm used to split the document into tokens (think
            words). '''

        return re_tokenize(string)

    def stem (self, string) :
        ''' This is a proxy to the specific stemming/lemmatisation algorithm used to map multiple forms of a word,
            e.g., diffrent capitlization, whitespace, or inflections, to a single form. '''

        return stem.clean(string)

    def gram (self, tokens) :
        gram_tuple = tuple(str(token) for token in tokens)
        if not self.is_stop_seq(gram_tuple) :
            return Gram(gram_tuple, prevalence=0)

    def word (self, tokens) :
        word_string = tokens[0]
        if not self.is_stop_seq(word_string) :
            return Word(word_string, prevalence=0)

    def seq (self, tokens) :
        if len(tokens) == 1 :
            return self.word(tokens)
        else :
            return self.gram(tokens)

def __test__ (ut) :
    from nlplib.core.model import Document, Database
    from nlplib.core.process.concordance import documents_containing
    from nlplib.core.process.token import re_tokenize

    corpus = [("I'd just like to interject for a moment. What you're referring to as Linux, is in fact, GNU/Linux, or "
               "as I've recently taken to calling it, GNU plus Linux."),
              ('Linux is not an operating system unto itself, but rather another free component of a fully '
               'functioning GNU system made useful by the GNU corelibs, shell utilities and vital system components '
               'comprising a full OS as defined by POSIX.')]

    max_gram_length = 3

    db = Database()

    with db as session :
        for text in corpus :
            session.add(Document(text))

    with db as session :
        for document in session.access.all_documents() :
            pass

        print(Parse(document)())

if __name__ == '__main__' :
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())

