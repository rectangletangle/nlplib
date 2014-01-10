

from nlplib.exterior.scrape.wikipedia import gather_documents
from nlplib.core.process.index import Indexed
from nlplib.general.iterate import chunked
from nlplib.general import timing

@timing
def make_db (db, amount=100) :
    total = 0
    for chunk in chunked(enumerate(gather_documents(amount), total + 1), 10, trail=True) :
        with db as session :
            indexed = Indexed(session)
            for total, document in chunk :
                if len(document) :
                    session.add(document)
                    indexed.add(document)
                    print(total, ':', repr(document))
    return total

if __name__ == '__main__' :
    from nlplib.data import builtin_db
    make_db(builtin_db())

