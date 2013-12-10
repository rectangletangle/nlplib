

from nlplib.core.acquire.scrape.wikipedia import gather_documents
from nlplib.core.model import Document
from nlplib.data import builtin_db

__all__ = ['add_documents_from_wikipedia_to_builtin_db']

def add_documents_from_wikipedia_to_builtin_db (amount) :
    with builtin_db() as session :
        for document in gather_documents(amount) :
            if len(document) :
                session.add(document)
        print(len(session.access.all_documents()))

if __name__ == '__main__' :
    add_documents_from_wikipedia_to_builtin_db(100)

