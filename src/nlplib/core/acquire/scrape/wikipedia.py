

from datetime import datetime

from nlplib.core.acquire.scrape import WebScraper, scrape_simultaneously
from nlplib.core.process.token import split
from nlplib.core.model import Document

__all__ = ['WikipediaScraper', 'gather_documents']

class WikipediaScraper (WebScraper) :

    _random_page_url = r'http://en.wikipedia.org/wiki/Special:Random'

    def _could_not_open_url (self, function, url, exception) :
        pass

    def scrape (self, function, url=None) :
        if url is None :
            url = self._random_page_url

        super().scrape(function, url)

def _flatten (iterable) :
    if not isinstance(iterable, str) :
        for item in iterable :
            yield from _flatten(item)
    else :
        yield iterable

def _make_document (url, soup) :
    raw = ''.join(_flatten(tag.find_all(text=True, recursive=True) for tag in soup.find_all('p')))

    return Document(raw,
                    word_count=len(list(split(raw))), # Strange silent errors when the list function is removed.
                    title='wikipedia',
                    url=url,
                    created_on=datetime.now())

def gather_documents (total=1, at_a_time=20) :
    documents = []

    scraper = WikipediaScraper()

    scrape_simultaneously((lambda : scraper.scrape(lambda url, soup : documents.append(_make_document(url, soup)))
                           for _ in range(total)),
                          at_a_time)

    return documents

def __demo__ () :
    from pprint import pprint
    pprint(gather_documents(4))

if __name__ == '__main__' :
    __demo__()

