

from datetime import datetime
from itertools import count

from nlplib.exterior.scrape.parse import parse_html
from nlplib.general.scrape import Scraped
from nlplib.core.process.token import split
from nlplib.core.model import Document
from nlplib.general.iterate import flattened

__all__ = ['RandomlyScrapedFromWikipedia', 'gather_documents']

class RandomlyScrapedFromWikipedia (Scraped) :

    url = 'https://en.wikipedia.org/wiki/Special:Random'

    def __init__ (self, amount=1, silent=True, *args, **kw) :

        counter = count() if amount in {'inf', float('inf'), -1} else range(amount)

        urls = (self.url for _ in counter)

        super().__init__(urls, silent=silent, *args, **kw)

def _make_document_from_response (response) :
    soup = parse_html(response.text)

    string = ''.join(flattened((tag.find_all(text=True, recursive=True) for tag in soup.find_all('p')),
                               basecase=lambda tags : isinstance(tags, str)))

    return Document(string,
                    word_count=len(list(split(string))),
                    title='wikipedia',
                    url=response.url,
                    created_on=datetime.now())

def gather_documents (*args, **kw) :
    for response in RandomlyScrapedFromWikipedia(*args, **kw) :
        yield _make_document_from_response(response)

def __demo__ () :
    for document in gather_documents('inf') :
        print(document.url + '\n' + repr(document), end='\n\n')

if __name__ == '__main__' :
    __demo__()

