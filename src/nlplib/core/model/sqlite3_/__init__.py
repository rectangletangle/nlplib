

from nlplib.core.model import abstract

class Database (abstract.Database) :
    raise NotImplementedError("The SQLite 3 back-end hasn't been implemented yet, this package depends on SQLAlchemy "
                              'for now.')

