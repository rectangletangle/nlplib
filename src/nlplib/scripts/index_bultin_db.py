

from nlplib.core.model import Indexer, Access
from nlplib.data import builtin_db

def index_builtin_db () :
    db = builtin_db()
    with db as session :
        indexer = Indexer(session)
        for i, document in enumerate(Access(session).all_documents()) :
            indexer.add(document)
            print(i, ':', repr(document))

if __name__ == '__main__' :
    index_builtin_db()

