

from nlplib.core.exc import NLPLibError

__all__ = ['ModelError', 'StorageError', 'IntegrityError']

class ModelError (NLPLibError) :
    pass

class StorageError (ModelError) :
    pass

class IntegrityError (ModelError) :
    pass

# todo : unit testing

