

__all__ = []

def gram_tuples (tokens, min_gram=2, max_gram=None) :
    # todo : Give a more descriptive name.

    amount_of_tokens = len(tokens)

    if max_gram is None :
        max_gram = amount_of_tokens

    assert min_gram >= 0
    assert max_gram >= 0
    assert max_gram >= min_gram

    for i in range(amount_of_tokens) :
        for j in range(i + min_gram, min(amount_of_tokens, i + max_gram) + 1) :
            yield tuple(tokens[i:j])

def __test__ (ut) :
    tokens = ["I'd", 'just', 'like', 'to', 'interject']

    ut.assert_equal(list(gram_tuples(tokens, min_gram=2, max_gram=None)),
                    [("I'd", 'just'),
                     ("I'd", 'just', 'like'),
                     ("I'd", 'just', 'like', 'to'),
                     ("I'd", 'just', 'like', 'to', 'interject'),
                     ('just', 'like'),
                     ('just', 'like', 'to'),
                     ('just', 'like', 'to', 'interject'),
                     ('like', 'to'),
                     ('like', 'to', 'interject'),
                     ('to', 'interject')])

    ut.assert_equal(list(gram_tuples(tokens, min_gram=1, max_gram=3)),
                    [("I'd",), ("I'd", 'just'), ("I'd", 'just', 'like'), ('just',), ('just', 'like'),
                     ('just', 'like', 'to'), ('like',), ('like', 'to'), ('like', 'to', 'interject'), ('to',),
                     ('to', 'interject'), ('interject',)])

    ut.assert_equal(list(gram_tuples(tokens, min_gram=0, max_gram=2)),
                    [(), ("I'd",), ("I'd", 'just'), (), ('just',), ('just', 'like'), (), ('like',), ('like', 'to'), (),
                     ('to',), ('to', 'interject'), (), ('interject',)])

    ut.assert_equal(list(gram_tuples(tokens, min_gram=2, max_gram=2)),
                    [("I'd", 'just'), ('just', 'like'), ('like', 'to'), ('to', 'interject')])

if __name__ == '__main__' :
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())

