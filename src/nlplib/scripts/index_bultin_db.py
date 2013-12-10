

from nlplib.general.time import timing
from nlplib.core.index import Indexed

def index_db (db) :
    total = 0
    with db as session :
        indexed = Indexed(session)
        for total, document in enumerate(session.access.all_documents(), 1) :
            indexed.add(document)
            print(total, ':', repr(document))

    return total

if __name__ == '__main__' :
    from nlplib.data import builtin_db
    number_of_documents = timing(index_db, log=lambda *args : print(*args, end=''))(builtin_db())
    print(' for %d document(s)' % number_of_documents)

