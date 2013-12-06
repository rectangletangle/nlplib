

__all__ = ['count']

def count (seq) :
    try :
        return seq.count
    except AttributeError :
        return 1

