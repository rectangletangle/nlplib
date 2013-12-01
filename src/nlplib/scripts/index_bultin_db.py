

from nlplib.data import builtin_db
from nlplib.core.index import Indexer

def index_builtin_db () :
    db = builtin_db()
    with db as session :
        indexer = Indexer(session)
        for i, document in enumerate(session.access.all_documents()) :
            indexer.add(document)
            print(i, ':', repr(document))

if __name__ == '__main__' :
    index_builtin_db()

