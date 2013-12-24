''' This storage back-end depends on the SQLAlchemy package
    url : http://www.sqlalchemy.org '''


from contextlib import contextmanager

from sqlalchemy.orm import sessionmaker
from sqlalchemy import exc as sqlalchemy_exc
from sqlalchemy import create_engine

from nlplib.core.model.sqlalchemy.map import default_mapped
from nlplib.core.model.sqlalchemy.access import Access
from nlplib.core.model.exc import IntegrityError, StorageError
from nlplib.core.model import abstract

__all__ = ['Session', 'Database']

class Session (abstract.Session) :
    def __init__ (self, sqlalchemy_session) :
        self._sqlalchemy_session = sqlalchemy_session

        self.access = Access(self)

    def add (self, object) :
        try :
            self._sqlalchemy_session.add(object)
        except sqlalchemy_exc.IntegrityError as exc :
            raise IntegrityError(str(exc))
        else :
            return object

    def add_many (self, objects) :
        objects = list(objects)

        try :
            self._sqlalchemy_session.add_all(objects)
        except sqlalchemy_exc.IntegrityError as exc :
            raise IntegrityError(str(exc))
        else :
            return objects

    def remove (self, object) :
        self._sqlalchemy_session.delete(object)

class Database (abstract.Database) :
    def __init__ (self, *args, **kw) :
        super().__init__(*args, **kw)

        self._sqlalchemy_engine = create_engine(self.path)
        self._make_sqlalchemy_session = sessionmaker(bind=self._sqlalchemy_engine)

        default_mapped.metadata.create_all(self._sqlalchemy_engine)

    @contextmanager
    def _make_session (self) :
        sqlalchemy_session = self._make_sqlalchemy_session()
        try :
            yield Session(sqlalchemy_session)
            sqlalchemy_session.commit()
        except sqlalchemy_exc.SQLAlchemyError as exc :
            sqlalchemy_session.rollback()
            raise StorageError(str(exc))
        finally :
            sqlalchemy_session.close()

def __test__ (ut) :
    from nlplib.core.model.abstract import abstract_test

    abstract_test(ut, Database)

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

