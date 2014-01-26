''' Infinitely yield random pages from Wikipedia. '''


from nlplib.exterior.scrape.wikipedia import RandomlyScrapedFromWikipedia
from nlplib import scraper

@scraper(silent=True)
def wikipedia_loop () :
    while True :
        yield RandomlyScrapedFromWikipedia.url

if __name__ == '__main__' :
    for i, response in enumerate(wikipedia_loop(), 1) :
        print(i, ':', repr(response))

