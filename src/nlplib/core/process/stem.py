''' This module is for stemming and lemmatisation algorithms, or other algorithms that act in a similar fashion to
    reduce string dimensionality. '''


__all__ = ['clean']

def clean (string) :
    ''' This acts as a very simple way to standardize, i.e., "clean up", a string. This is done by making all
        whitespace into single spaces and making all characters lowercase. If there are multiple consecutive whitespace
        characters, they'll be made into a single space character.'''

    return ' '.join(str(string).split()).strip().lower()

def __test__ (ut) :
    ut.assert_equal(clean('And  the \n\tcat aTe\tthe\n\nsandwich'), 'and the cat ate the sandwich')
    ut.assert_equal(clean('and the cat ate'), 'and the cat ate')
    ut.assert_equal(clean('andTdsad'), 'andtdsad')
    ut.assert_equal(clean('dsfasdfa'), 'dsfasdfa')

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

