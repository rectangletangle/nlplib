''' This package contains classes that act as abstract base classes for the various persistence back-ends. '''

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

    def _remove (self, object) :
        raise NotImplementedError

    def remove (self, object) :
        if hasattr(object, '_associated') :
            associated_objects = object._associated(self)

            for associated_object in associated_objects :
                self._remove(associated_object)

        self._remove(object)

class Database (Base) :
    ''' This class represents a database. Generally you don't interface with the database directly so much, but instead
        with a database session object. '''

    def __init__ (self, path=r'sqlite:///:memory:') :
        self.path = path

        self._sessions_currently_open = []

    def __repr__ (self, *args, **kw) :
        return super().__repr__(self.path, *args, **kw)

    def __enter__ (self) :
        session = self.session()
        self._sessions_currently_open.append(session)
        return session.__enter__()

    def __exit__ (self, *args, **kw) :
        try :
            return self._sessions_currently_open.pop().__exit__(*args, **kw)
        except IndexError :
            pass

    def __call__ (self, function) :
        ''' This allows a database instance to be used as a decorator. '''

        @wraps(function)
        def within_session (*args, **kw) :
            with self.session() as session :
                return function(session, *args, **kw)

        return within_session

    @contextmanager
    def session (self) :
        ''' This allows for objects within the database to be accessed and manipulated within a transaction like
            context. '''

        raise NotImplementedError

def abstract_test (ut, db_cls) :
    from nlplib.core.model import Word

    db = db_cls()

    with db as session :
        session.add(Word('a'))
        session.add(Word('b'))
        session.add(Word('c'))
        session.add(Word(b'\xc3\x9c'.decode()))

    with db as session :
        session.remove(session.access.word('b'))

    with db as session :
        ut.assert_equal(sorted(session.access.all_words()), [Word('a'), Word('c'), Word(b'\xc3\x9c'.decode())])

    db = db_cls()

    with db as session_0 :
        session_0.add(Word('x'))
        with db as session_1 :
            session_1.add(Word('y'))
            session_0.add(Word('z'))

    with db as session :
        ut.assert_equal(sorted(session.access.all_words()), [Word('x'), Word('y'), Word('z')])

    db = db_cls()

    @db
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

    # Tests rollback.
    db = db_cls()
    try :
        with db as session :
            session.add(Word('a'))
            session.add(Word('b'))
            raise IOError # An arbitrary exception
    except IOError :
        pass

    with db as session :
        ut.assert_equal(list(session.access.all_words()), [])

    @db
    def bar (session) :
        session.add(Word('c'))
        session.add(Word('d'))
        raise KeyError

    try :
        bar()
    except KeyError :
        pass

    with db as session :
        ut.assert_equal(list(session.access.all_words()), [])


