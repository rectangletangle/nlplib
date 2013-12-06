

from nlplib.data import builtin_db
from nlplib.core.index import Indexed

def index_builtin_db () :
    db = builtin_db()
    with db as session :
        indexed = Indexed(session)
        for i, document in enumerate(session.access.all_documents()) :
            indexed.add(document)
            print(i, ':', repr(document))

if __name__ == '__main__' :
    index_builtin_db()

