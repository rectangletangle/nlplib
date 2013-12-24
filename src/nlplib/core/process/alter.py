''' This module is for creating similar, yet altered versions of a sequence. '''


from nlplib.core.process import token

__all__ = ['deletions', 'transpositions', 'replacements', 'insertions', 'alterations', 'similar']

def deletions (halves) :
    for pre, post in halves :
        with_deletion = pre + post[1:]

        # Checking the length of <post> makes sure that we're actually deleting something, and checking the length of
        # <with_deletion> makes sure we're not yielding an empty sequence (which could happen if len(sequence) <= 1)
        if len(post) and len(with_deletion) :
            yield with_deletion

def transpositions (halves) :
    for pre, post in halves :
        if len(post) > 1 :
            yield pre + post[1] + post[0] + post[2:]

def replacements (halves, replacements) :
    for pre, post in halves :
        for replacement in replacements :
            if len(post) :
                yield pre + replacement + post[1:]

def insertions (halves, insertions) :
    for pre, post in halves :
        for insertion in insertions :
            yield pre + insertion + post

def alterations (seq, introduce) :
    halves = list(token.map_over_indexes(token.halve, seq))
    return set().union(deletions(halves),
                       transpositions(halves),
                       replacements(halves, introduce),
                       insertions(halves, introduce))

def similar (seq, introduce, changes=2) :
    ''' This generates similar sequences to the sequence given. For example, a word would generate strings with
        characters inserted, replaced, transposed (switched order), and deleted. This function also works at the gram
        level, by switching out the words within the gram.

        seq : The sequence from which new similar sequences will be made.
        introduce : Characters, words, or grams that will be added.
        changes : The amount of changes the most dissimilar sequence generated will be from the given sequence. '''

    if changes :
        altered_seqs = alterations(seq, introduce)
        for altered_seq in tuple(altered_seqs) :
            altered_seqs.update(similar(altered_seq, introduce, changes-1))
        return altered_seqs
    else :
        return set()

def __test__ (ut) :
    from nlplib.core.model import Seq, Gram, Word

    introduce = 'abc'

    # Test an odd length string.
    seq = 'hello'
    halves = list(token.map_over_indexes(token.halve, seq))

    ut.assert_equal(list(deletions(halves)), ['ello', 'hllo', 'helo', 'helo', 'hell'])
    ut.assert_equal(list(transpositions(halves)), ['ehllo', 'hlelo', 'hello', 'helol'])
    ut.assert_equal(list(replacements(halves, introduce)),
                    ['aello', 'bello', 'cello', 'hallo', 'hbllo', 'hcllo', 'healo', 'heblo', 'heclo', 'helao', 'helbo',
                     'helco', 'hella', 'hellb', 'hellc'])
    ut.assert_equal(list(insertions(halves, introduce)),
                    ['ahello', 'bhello', 'chello', 'haello', 'hbello', 'hcello', 'heallo', 'hebllo', 'hecllo',
                     'helalo', 'helblo', 'helclo', 'hellao', 'hellbo', 'hellco', 'helloa', 'hellob', 'helloc'])
    ut.assert_equal(alterations(seq, introduce),
                    {'heblo', 'heallo', 'hellb', 'hellco', 'ehllo', 'hellao', 'hbello', 'hecllo', 'bello', 'helol',
                     'helao', 'hallo', 'hlelo', 'helblo', 'hell', 'ello', 'helloa', 'helo', 'helclo', 'heclo',
                     'chello', 'cello', 'haello', 'hellc', 'aello', 'hella', 'hellbo', 'hcllo', 'hello', 'helbo',
                     'hbllo', 'helalo', 'hcello', 'hellob', 'helco', 'bhello', 'hllo', 'helloc', 'hebllo', 'healo',
                     'ahello'})

    # Test an even length string.
    seq = 'fish'
    halves = list(token.map_over_indexes(token.halve, seq))

    ut.assert_equal(list(deletions(halves)), ['ish', 'fsh', 'fih', 'fis'])
    ut.assert_equal(list(transpositions(halves)), ['ifsh', 'fsih', 'fihs'])
    ut.assert_equal(list(replacements(halves, introduce)),
                    ['aish', 'bish', 'cish', 'fash', 'fbsh', 'fcsh', 'fiah', 'fibh', 'fich', 'fisa', 'fisb', 'fisc'])
    ut.assert_equal(list(insertions(halves, introduce)),
                    ['afish', 'bfish', 'cfish', 'faish', 'fbish', 'fcish', 'fiash', 'fibsh', 'ficsh', 'fisah', 'fisbh',
                     'fisch', 'fisha', 'fishb', 'fishc'])
    ut.assert_equal(alterations(seq, introduce),
                    {'fcsh', 'aish', 'fsih', 'ifsh', 'fis', 'faish', 'fbsh', 'fih', 'fisbh', 'fibh', 'fisch', 'fishb',
                     'bish', 'afish', 'bfish', 'fbish', 'fsh', 'fishc', 'fcish', 'fisha', 'fibsh', 'ish', 'fiah',
                     'cfish', 'fisah', 'fich', 'fisb', 'fisc', 'fash', 'fisa', 'fiash', 'fihs', 'cish', 'ficsh'})

    for hello in ['hello', Seq('hello'), Word('hello')] :
        ut.assert_equal(similar(hello, 'ab', 1),
                        {'haello', 'helblo', 'helbo', 'hallo', 'hellob', 'helo', 'helloa', 'healo', 'hell', 'helalo',
                         'hbllo', 'hellbo', 'ello', 'hella', 'hellb', 'hello', 'aello', 'heallo', 'ahello', 'hbello',
                         'bhello', 'bello', 'helol', 'hllo', 'helao', 'hebllo', 'heblo', 'hellao', 'ehllo', 'hlelo'})

    ut.assert_equal(similar('h', 'ab', 1), {'hb', 'ha', 'ah', 'a', 'b', 'bh'})

    # Test with mixed types introduced.
    correct_output = {Gram('hello a'), Gram('b'), Gram('a'), Gram('hello b'), Gram('a hello'), Gram('b hello')}

    for introduce in ['ab', ['a', 'b'], [Seq('a'), Seq('b')], [Word('a'), Word('b')], [Gram('a'), Gram('b')]] :
        ut.assert_equal(similar(Gram('hello'), introduce, 1), correct_output)

    for a in ['a', Seq('a'), Gram('a'), Word('a')] :
        for b in ['b', Seq('b'), Gram('b'), Word('b')] :
            ut.assert_equal(similar(Gram('hello'), [a, b], 1), correct_output)

    for introduce in [['ab'], [Seq('ab')], [Gram('ab')], [Word('ab')]] :
        ut.assert_equal(similar(Gram('hello'), introduce, 1), {Gram('ab'), Gram('ab hello'), Gram('hello ab')})

    # Test <similar> using the words inside a gram.
    ut.assert_equal(similar(Gram('a b'), Gram('c d'), 1),
                    {Gram('d a b'), Gram('d b'), Gram('a c'), Gram('a'), Gram('b'), Gram('a d b'), Gram('a d'),
                     Gram('a b d'), Gram('b a'), Gram('a c b'), Gram('c a b'), Gram('c b'), Gram('a b c')})

    # Tests <similar> using whole grams.
    ut.assert_equal(similar(Gram('a b'), [Gram('c d'), Gram('e f')], 1),
                    {Gram('a c d'), Gram('a c d b'), Gram('b a'), Gram('c d a b'), Gram('a b e f'), Gram('c d b'),
                     Gram('a b c d'), Gram('a e f'), Gram('b'), Gram('a e f b'), Gram('a'), Gram('e f a b'),
                     Gram('e f b')})

    # Tests deep changes using <similar>.
    ut.assert_equal(similar('a', 'bc', 2),
                    {'bab', 'bac', 'cba', 'a', 'cab', 'c', 'b', 'ba', 'acc', 'acb', 'cac', 'abc', 'cca', 'abb', 'bca',
                     'bba', 'ac', 'ab', 'ca', 'bc', 'cc', 'cb', 'bb'})

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

