''' This package contains classes that act as abstract base classes for the various storage back-ends. '''

from functools import wraps
from contextlib import contextmanager

from nlplib.core.base import Base

__all__ = ['Session', 'Database', 'abstract_test']

class Session (Base) :
    ''' This class represents a session with a database. Within the session context, objects can be added, removed, or
        queried (using the session dependent access class). '''

    def __init__ (self) :
        self.access = None

    def add (self, object) :
        raise NotImplementedError

    def add_many (self, objects) :
        return [self.add(object) for object in objects]

    def remove (self, object) :
        raise NotImplementedError

class Database (Base) :
    ''' This class represents a database. Generally you don't interface with the database directly too much, but
        instead with a database session object. '''

    def __init__ (self, path=r'sqlite:///:memory:') :
        self.path = path

        self._sessions_currently_open = []

    def __repr__ (self, *args, **kw) :
        return super().__repr__(self.path, *args, **kw)

    def __enter__ (self) :
        session = self._make_session()
        self._sessions_currently_open.append(session)
        return session.__enter__()

    def __exit__ (self, *args, **kw) :
        try :
            return self._sessions_currently_open.pop().__exit__(*args, **kw)
        except IndexError :
            pass

    @contextmanager
    def _make_session (self) :
        raise NotImplementedError

    def session (self, function=None) :
        ''' This allows for objects within the database to be accessed and manipulated within a transaction like
            context. '''

        if callable(function) :
            @wraps(function)
            def with_session (*args, **kw) :
                with self._make_session() as session :
                    return function(session, *args, **kw)
            return with_session
        else :
            return self._make_session()

def abstract_test (ut, db_cls) :
    from nlplib.core.model import Word

    db = db_cls()

    with db as session :
        session.add(Word('a'))
        session.add(Word('b'))
        session.add(Word('c'))

    with db as session :
        session.remove(session.access.word('b'))

    with db as session :
        ut.assert_equal(sorted(session.access.all_words()), [Word('a'), Word('c')])

    db = db_cls()

    with db as session_0 :
        session_0.add(Word('x'))
        with db as session_1 :
            session_1.add(Word('y'))
            session_0.add(Word('z'))

    with db as session :
        ut.assert_equal(sorted(session.access.all_words()), [Word('x'), Word('y'), Word('z')])

    db = db_cls()

    @db.session
    def foo (session, *args) :
        for arg in args :
            session.add(Word(arg))

    foo('0', '1', '2')

    with db as session :
        ut.assert_equal(sorted(session.access.all_words()), [Word('0'), Word('1'), Word('2')])

    db = db_cls()

    with db.session() as session :
        session.add_many(Word(number) for number in '345')

    with db.session() as session :
        ut.assert_equal(sorted(session.access.all_words()), [Word('3'), Word('4'), Word('5')])

