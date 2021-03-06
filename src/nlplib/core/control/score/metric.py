''' Various scoring metrics. '''


__all__ = ['count', 'levenshtein_distance']

def count (seq, default=1) :
    ''' Score a sequence based on how many times it has been encountered. '''

    try :
        return seq.count if not callable(seq.count) else default
    except AttributeError :
        return default

def levenshtein_distance (seq_0, seq_1) :
    ''' This calculates the Levenshtein distance between two sequences (usually strings). '''

    seq_0_len = len(seq_0)
    seq_1_len = len(seq_1)

    if seq_0_len > seq_1_len :
        seq_1, seq_0 = seq_0, seq_1

    distances = tuple(range(seq_0_len + 1))
    for i, char_1 in enumerate(seq_1) :
        new_distances = (i+1,)

        for j, char_0 in enumerate(seq_0) :

            if char_0 == char_1 :
                new_distances += (distances[j],)
            else :
                min_distance = min((distances[j], distances[j + 1], new_distances[-1]))
                new_distances += (min_distance+1,)

        distances = new_distances

    return distances[-1]

def __test__ (ut) :
    from nlplib.general.unittest import mock

    ut.assert_equal(count('foo'), 1)
    ut.assert_equal(count('bar', default=-3), -3)
    ut.assert_equal(count(mock(count=35)), 35)

    ut.assert_equal(levenshtein_distance('hello', 'hello'), 0)
    ut.assert_equal(levenshtein_distance('hello', 'helo'), 1)
    ut.assert_equal(levenshtein_distance('hello', 'helo'), 1)
    ut.assert_equal(levenshtein_distance([0, 1, 2], [0, 1, 2]), 0)
    ut.assert_equal(levenshtein_distance([0, 1, 2], [0, 2]), 1)

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

