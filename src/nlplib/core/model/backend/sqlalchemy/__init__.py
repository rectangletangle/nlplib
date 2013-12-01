''' This storage back-end depends on the SQLAlchemy package
    url : http://www.sqlalchemy.org '''


from contextlib import contextmanager

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from nlplib.core.model.backend.sqlalchemy.map import default_mapper
from nlplib.core.model.backend.sqlalchemy.access import Access
from nlplib.core.model.backend import abstract

__all__ = ['Session', 'Database']

class Session (abstract.Session) :
    def __init__ (self, sqlalchemy_session) :
        self._sqlalchemy_session = sqlalchemy_session

        self.access = Access(self)

    def add (self, object) :
        self._sqlalchemy_session.add(object)
        self._sqlalchemy_session.flush((object,)) # todo : make lazy

        return object

    def _as_dict (self, table, object) :
        return {attr_name : getattr(object, attr_name) for attr_name in table.c.keys()}

    def add_many (self, objects) :
        objects = list(objects)

        try :
            table = default_mapper.classes_with_tables[objects[0].__class__]
        except IndexError :
            pass
        else :
            self._sqlalchemy_session.execute(table.insert(), [self._as_dict(table, object) for object in objects])

        return objects

    def remove (self, object) :
        self._sqlalchemy_session.delete(object)

class Database (abstract.Database) :
    def __init__ (self, *args, **kw) :
        super().__init__(*args, **kw)

        self._sqlalchemy_engine = create_engine(self.path)
        self._make_sqlalchemy_session = sessionmaker(bind=self._sqlalchemy_engine)

        default_mapper.metadata.create_all(self._sqlalchemy_engine)

    @contextmanager
    def session (self) :
        sqlalchemy_session = self._make_sqlalchemy_session()
        try :
            yield Session(sqlalchemy_session)
            sqlalchemy_session.commit()
        except :
            sqlalchemy_session.rollback()
            raise
        finally :
            sqlalchemy_session.close()

def __test__ (ut) :
    from nlplib.core.model.backend.abstract import abstract_test

    abstract_test(ut, Database)

if __name__ == '__main__' :
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())

