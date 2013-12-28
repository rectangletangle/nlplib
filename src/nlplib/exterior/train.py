

from nlplib.core.process.parse import Parsed
from nlplib.general.iterate import windowed

# todo : __all__

def is_readable (seqs, threshold=0.5, unknown=None) :
    seqs = tuple(seqs)
    return (seqs.count(unknown) / len(seqs)) <= threshold

def _known_or_unkown (known, seqs, unknown=None) :
    for seq in seqs :
        try :
            yield known[seq]
        except KeyError :
            yield unknown

def usable (known, documents, is_usable=is_readable, gram_size=4, split_index=None) :
    known = {value : value for value in known}

    if split_index is None :
        split_index = gram_size - 1

    for document in documents :
        for gram in windowed(Parsed(document), gram_size) :
            input_ = tuple(_known_or_unkown(known, gram[:split_index]))
            correct_output = tuple(_known_or_unkown(known, gram[split_index:]))

            if is_usable(input_) and is_usable(correct_output) :
                yield (input_, correct_output)

def __test__ (ut) :
    from nlplib.core.model import Database, Word

    ut.assert_true(not is_readable([None, None]))
    ut.assert_true(is_readable(['', None]))
    ut.assert_true(is_readable(['', '']))

    document = '''Python is a widely used general-purpose, high-level programming language. Its design
                  philosophy emphasizes code readability, and its syntax allows programmers to express concepts in
                  fewer lines of code than would be possible in languages such as C. The language provides
                  constructs intended to enable clear programs on both a small and large scale. '''

    db = Database()

    with db as session :
        for string in 'is a widely used than be possible to enable clear programs both small and large'.split() :
            session.add(Word(string))

    def correct () :
        yield ((None, Word('is')),              (Word('a'),))
        yield ((Word('is'), Word('a')),         (Word('widely'),))
        yield ((Word('a'), Word('widely')),     (Word('used'),))
        yield ((Word('than'), None),            (Word('be'),))
        yield ((None, Word('be')),              (Word('possible'),))
        yield ((None, Word('to')),              (Word('enable'),))
        yield ((Word('to'), Word('enable')),    (Word('clear'),))
        yield ((Word('enable'), Word('clear')), (Word('programs'),))
        yield ((Word('programs'), None),        (Word('both'),))
        yield ((None, Word('both')),            (Word('a'),))
        yield ((Word('both'), Word('a')),       (Word('small'),))
        yield ((Word('a'), Word('small')),      (Word('and'),))
        yield ((Word('small'), Word('and')),    (Word('large'),))

    with db as session :
        for io, correct_io in zip(usable(set(session.access.all_seqs()), [document], gram_size=3),
                                  correct()) :

            ut.assert_equal(io, correct_io)

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

