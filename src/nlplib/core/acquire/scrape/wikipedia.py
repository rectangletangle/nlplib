

from datetime import datetime

from nlplib.core.acquire.scrape.parse import parse_html
from nlplib.core.acquire.scrape import Scraped
from nlplib.core.process.token import split
from nlplib.core.model import Document
from nlplib.general.thread import simultaneously

__all__ = ['RandomlyScrapedFromWikipedia', 'gather_documents']

class RandomlyScrapedFromWikipedia (Scraped) :
    def __init__ (self, amount=1, silent=True, *args, **kw) :
        urls = (r'http://en.wikipedia.org/wiki/Special:Random' for _ in range(amount))
        super().__init__(urls, silent=silent, *args, **kw)

def _flatten (iterable) :
    if not isinstance(iterable, str) :
        for item in iterable :
            yield from _flatten(item)
    else :
        yield iterable

def _make_document_from_response (response) :
    soup = parse_html(response.text)

    raw = ''.join(_flatten(tag.find_all(text=True, recursive=True) for tag in soup.find_all('p')))

    return Document(raw,
                    word_count=len(list(split(raw))),
                    title='wikipedia',
                    url=response.url,
                    created_on=datetime.now())

def gather_documents (*args, **kw) :
    for response in RandomlyScrapedFromWikipedia(*args, **kw) :
        yield _make_document_from_response(response)

def __demo__ () :
    print('\n\n'.join(document.url + '\n' + repr(document) for document in gather_documents(4)))

if __name__ == '__main__' :
    __demo__()

