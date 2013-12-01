

from nlplib.data import builtin_db

def index_builtin_db () :
    db = builtin_db()
    with db as session :
        for i, document in enumerate(session.access.all_documents()) :
            session.index.add(document)
            print(i, ':', repr(document))

if __name__ == '__main__' :
    index_builtin_db()

