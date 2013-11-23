

__all__ = ['prevalence']

def prevalence (seq) :
    try :
        return seq.prevalence
    except AttributeError :
        return 1

