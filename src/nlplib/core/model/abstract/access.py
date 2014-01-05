

from nlplib.core.process.token import split
from nlplib.core.model import SessionDependent, Document, Seq, Gram, Word, Index, NeuralNetwork, IONode, Node, Link

__all__ = ['Access', 'abstract_test']

class Access (SessionDependent) :
    ''' This class contains methods which act as convenient abstractions of common database queries; this provides the
        primary way to access the objects stored within a database. '''

    def words (self, string, splitter=split) :
        ''' This returns the word objects corresponding to the word substrings within a string. If no word object is
            found for a particular substring, <None> is used. '''

        if isinstance(string, str) :
            return [self.word(word_string) for word_string in splitter(string)]
        else :
            # Treat string as a collection of strings.
            return [self.word(word_string) for word_string in string]

    def vocabulary (self) :
        return self.all_words()

    def corpus (self) :
        # todo : remove if corpus model is added
        return self.all_documents()

    def _all (self, cls, chunk_size=100) :
        raise NotImplementedError

    def all_documents (self, *args, **kw) :
        ''' This returns all of the document objects in the database. '''

        return self._all(Document, *args, **kw)

    def all_seqs (self, *args, **kw) :
        return self._all(Seq, *args, **kw)

    def all_grams (self, *args, **kw) :
        ''' This returns all of the gram objects in the database. '''

        return self._all(Gram, *args, **kw)

    def all_words (self, *args, **kw) :
        ''' This returns all of the word objects in the database. '''

        return self._all(Word, *args, **kw)

    def all_indexes (self, *args, **kw) :
        return self._all(Index, *args, **kw)

    def all_neural_networks (self, *args, **kw) :
        return self._all(NeuralNetwork, *args, **kw)

    def all_nodes (self, *args, **kw) :
        return self._all(Node, *args, **kw)

    def all_io_nodes (self, *args, **kw) :
        return self._all(IONode, *args, **kw)

    def all_links (self, *args, **kw) :
        return self._all(Link, *args, **kw)

    def _seq (self, cls, string) :
        raise NotImplementedError

    def seq (self, string) :
        return self._seq(Seq, string)

    def gram (self, gram_string_or_tuple) :
        ''' This returns the gram object corresponding to a string or tuple, or <None>.
            gram_string = 'the cat ate'
            gram_tuple  = ('the', 'cat', 'ate') '''

        return self._seq(Gram, str(Gram(gram_string_or_tuple)))

    def word (self, word_string) :
        ''' This returns the word object corresponding to a string, or <None>. '''

        return self._seq(Word, str(word_string))

    def specific (self, cls, id) :
        ''' This returns a specific object by id. '''

        raise NotImplementedError

    def most_common (self, cls=None, top=10) :
        ''' This returns most common objects based on their count. '''

        raise NotImplementedError

    def indexes (self, document) :
        ''' This returns all of the indexes (with their sequences) referencing the document. '''

        raise NotImplementedError

    def matching (self, strings, cls=Seq, chunk_size=100) :
        ''' This returns sequences (grams and words) that match the given list of strings.

            Note : This method is typically implemented using the SQL <IN> operator. Some database systems have
            stipulations regarding the maximum size of the set used for membership testing. The optional <chunk_size>
            argument allows the set to be broken up into multiple smaller sets (chunks), with a length corresponding to
            <chunk_size>, so that the set may fall under this limit. '''

        raise NotImplementedError

    def neural_network (self, name) :
        raise NotImplementedError

    def nn (self, *args, **kw) :
        return self.neural_network(*args, **kw)

    def nodes_for_seqs (self, seqs, input=None) :
        raise NotImplementedError

    def input_nodes_for_seqs (self, *args, **kw) :
        return self.nodes_for_seqs(*args, input=True, **kw)

    def output_nodes_for_seqs (self, *args, **kw) :
        return self.nodes_for_seqs(*args, input=False, **kw)

    def link (self, neural_network, input_node, output_node) :
        raise NotImplementedError

def abstract_test (ut, db_cls) :

    from nlplib.core.model import Seq, Gram, Word, Document

    chars = 'abc'

    db = db_cls()

    with db as session :
        document = session.add(Document('foo'))
        for count, char in enumerate(chars, 1) :
            for cls in (Seq, Gram, Word) :
                seq = session.add(cls(char))
                seq.indexes.extend([Index(document, None, None, None, None) for _ in range(count)])

    def mock (classes, chars) :
        return sorted(cls(char) for char in chars for cls in classes)

    with db as session :
        ut.assert_equal(sorted(session.access.all_seqs()),
                        mock((Seq, Gram, Word), chars))
        ut.assert_equal(sorted(session.access.all_grams()),
                        mock((Gram,), chars))
        ut.assert_equal(sorted(session.access.all_words()),
                        mock((Word,), chars))
        ut.assert_equal(sorted(session.access.vocabulary()),
                        sorted(session.access.all_words()))
        ut.assert_equal(sorted(session.access.corpus(), key=str),
                        sorted(session.access.all_documents(), key=str))

        ut.assert_equal(sorted(session.access.most_common(cls=Seq, top=3)),
                        mock((Seq, Gram, Word), 'c'))

        ut.assert_equal(sorted(session.access.most_common(cls=Word, top=2)),
                        mock((Word,), 'bc'))

        ut.assert_equal(session.access.word('a'), Word('a'))
        ut.assert_equal(session.access.word('z'), None)

        ut.assert_equal(session.access.words('b a c'), [Word('b'), Word('a'), Word('c')])
        ut.assert_equal(session.access.words('b z c'), [Word('b'), None, Word('c')])
        ut.assert_true(session.access.words('b a c') != [Word('a'), Word('b'), Word('c')])
        ut.assert_equal(session.access.words(''), [])

        ut.assert_equal(sorted(session.access.matching(['a', 'b'])), mock((Seq, Gram, Word), 'ab'))
        ut.assert_equal(sorted(session.access.matching(['a', 'b'], Word)), mock((Word,), 'ab'))
        ut.assert_equal(sorted(session.access.matching([])), [])

    db = db_cls()

    with db as session :
        session.add(NeuralNetwork('nn'))
        for i in range(3) :
            session.add(Word(str(i)))

    with db as session :
        nn = session.access.nn('nn')
        for i, word in enumerate(session.access.all_words()) :
            session.add(IONode(nn, word, i % 2 == 0))
        session.add(IONode(nn, None, (i + 1) % 2 == 0))

    with db as session :
        nn = session.access.nn('nn')

        def seqs (query, *args, **kw) :
            return {node.object for node in query(*args, **kw)}

        word = session.access.word

        ut.assert_equal(seqs(session.access.input_nodes_for_seqs, nn, [None, word('0'), word('1')]),
                        {Word('0')})
        ut.assert_equal(seqs(session.access.output_nodes_for_seqs, nn, [None, word('0'), word('1')]),
                        {None, Word('1')})
        ut.assert_equal(seqs(session.access.nodes_for_seqs, nn, [None, word('0'), word('1')]),
                        {None, Word('0'), Word('1')})

