

from nlplib.core.base import Base

class Model (Base) :
    ''' The base class for all models. '''

    pass

class SessionDependent (Base) :
    ''' A base class for classes which depend on a database session. '''

    def __init__ (self, session) :
        self.session = session

