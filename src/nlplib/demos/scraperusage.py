''' A demonstration of the web-scraper decorator. Because part of this demonstration depends on the internet, it's
    possible that this can throw exceptions. '''


import nlplib

@nlplib.scraper
def scrape_python_dot_org () :
    yield 'http://python.org'

# Because of the <silent> argument, this won't throw exceptions if something goes awry.
@nlplib.scraper(silent=True)
def some_scraped_stuff () :
    # Scrape from multiple diffrent URLs.
    yield 'http://python.org'
    yield 'http://wikipedia.org'
    yield 'http://github.com'

if __name__ == '__main__' :

    for response in scrape_python_dot_org() :
        # The URL of the response, this can be different from the original input URL if you end up getting redirected.
        print(response.url)

        # The first 100 characters of the string containing the response's text, typically HTML code.
        print(str(response)[:100] + '...')
        print()

    for response in some_scraped_stuff() :
        print(response.url)
        print(str(response)[:100] + '...')
        print()

