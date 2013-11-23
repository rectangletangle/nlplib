

from os import path

from nlplib.core.model import Database

__all__ = ['builtin_db']

def _src (file_path, name) :
    return path.join(path.split(file_path)[0], name)

def builtin_db (name='builtin.db', *args, **kw) :
    return Database(r'sqlite:///' + _src(__file__, name), *args, **kw)

def __demo__ () :
    from nlplib.core.model import Access, Word

    with builtin_db() as session :
        access = Access(session)
        print(len(access.all_documents()))
        for word in access.most_prevalent(Word, top=10) :
            print(repr(word))

if __name__ == '__main__' :
    __demo__()

