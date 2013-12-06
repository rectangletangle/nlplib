

from os import path

from nlplib.core.model import Database

__all__ = ['builtin_db']

def _src (file_path, name) :
    return path.join(path.split(file_path)[0], name)

def builtin_db (*args, name='builtin.db', db_cls=Database, **kw) :
    return db_cls(r'sqlite:///' + _src(__file__, name), *args, **kw)

def __demo__ () :
    from nlplib.core.model import Word

    with builtin_db() as session :
        print(len(session.access.all_documents()))
        for word in session.access.most_common(Word, top=10) :
            print(repr(word))

if __name__ == '__main__' :
    __demo__()

