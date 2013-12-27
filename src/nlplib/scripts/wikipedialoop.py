

from nlplib.exterior.scrape.wikipedia import RandomlyScrapedFromWikipedia
from nlplib.general.scrape import scraper

__all__ = ['wikipedia_loop']

@scraper(silent=True, chunk_size=1)
def wikipedia_loop () :
    while True :
        yield RandomlyScrapedFromWikipedia.url

if __name__ == '__main__' :
    for i, response in enumerate(wikipedia_loop(), 1) :
        print(i, ':', repr(response))

