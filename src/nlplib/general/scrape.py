''' Tools for scraping text from the internet. '''


from functools import wraps
from urllib.request import build_opener, URLError

from nlplib.general.represent import pretty_truncate
from nlplib.general.thread import simultaneously
from nlplib.general.iterate import chunked
from nlplib.core.base import Base
from nlplib.core.exc import NLPLibError

__all__ = ['CouldNotOpenURL', 'Response', 'Scraped', 'scraper']

class CouldNotOpenURL (NLPLibError) :
    pass

class Response (Base) :
    ''' A class representing a response from a website. '''

    __slots__ = ('url', 'text', 'request_url')

    def __init__ (self, url, text, request_url=None) :
        self.url = url
        self.text = str(text)

        self.request_url = request_url if request_url is not None else self.url

    def __repr__ (self, *args, **kw) :
        return super().__repr__(url=self.url, text=pretty_truncate(self.text), *args, **kw)

    def __str__ (self) :
        return self.text

    def __eq__ (self, other) :
        try :
            return (self.url, self.text, self.request_url) == (other.url, other.text, other.request_url)
        except AttributeError :
            False

    def __hash__ (self) :
        return hash((self.url, self.text, self.request_url))

class Scraped (Base) :
    ''' A base web scraper class. '''

    # todo : unify serial chunk_size and max_workers

    def __init__ (self, urls, silent=False, revisit=True, chunk_size=20, max_workers=None, serial=False,
                  user_agent='nlplib', encoding='utf-8', visited_urls=()) :

        self.urls = urls
        self.silent = silent
        self.revisit = revisit

        self.chunk_size  = chunk_size
        self.max_workers = max_workers
        self.serial      = serial

        self.user_agent = user_agent
        self.encoding = encoding

        self.visited_urls = set(visited_urls)

    def __iter__ (self) :
        self.visited_urls.clear()

        opener = self._make_opener()

        for urls in chunked(self.urls, self.chunk_size, trail=True) :
            responses = []

            scraping_functions = (lambda url=url : responses.append(self._scrape(opener, url))
                                  for url in urls)

            if not self.serial :
                simultaneously(scraping_functions, max_workers=self.max_workers)
            else :
                for function in scraping_functions :
                    function()

            for response in responses :
                if response is not None :
                    yield response

    def _make_opener (self) :
        opener = build_opener()
        opener.addheaders[:] = (('User-agent', self.user_agent),)
        return opener

    def _scrape (self, opener, request_url) :
        try :
            with opener.open(request_url) as page :
                response = Response(url=page.geturl(), text=page.read().decode(self.encoding), request_url=request_url)
                return self._could_open_url(response)
        except URLError as exc :
            return self.could_not_open_url(exc, request_url)

    def _could_open_url (self, response) :
        if response.url in self.visited_urls :
            return self.been_at_url_before(response)
        else :
            if not self.revisit :
                self.visited_urls.add(response.url)

            return self.not_been_at_url_before(response)

    def not_been_at_url_before (self, response) :
        ''' The behavior for when the scraper gets a response from a URL it hasn't seen before. '''

        return response

    def been_at_url_before (self, response) :
        ''' Behavior for when the scraper gets a response from a URL it has seen before. '''

        pass

    def could_not_open_url (self, exc, url) :
        ''' Behavior when a scraper couldn't get a response from a url. If the <silent> attribute is
            false (the default), this will throw errors. '''

        if not self.silent :
            raise CouldNotOpenURL(str(exc) + '\nhappened with the url : %s' % str(url))

def scraper (*args, cls=Scraped, **kw) :
    ''' This takes a generator function that yields URLs, and makes it yield responses from those URLs. The <Scraped>
        class has been specifically designed to allow for the use of infinite generator functions. '''

    def wrapper (generator) :
        @wraps(generator)
        def with_generator (*wrapped_args, **wrapped_kw) :
            return cls(generator(*wrapped_args, **wrapped_kw), *args, **kw)
        return with_generator
    return wrapper

def __test__ (ut) :
    from nlplib.general.unittest import mock

    urls = {'a' : mock(geturl=lambda : 'a', read=lambda : b'aaa'),
            'b' : mock(geturl=lambda : 'b', read=lambda : b'bbb'),
            'c' : mock(geturl=lambda : 'c', read=lambda : b'ccc'),
            'd' : mock(geturl=lambda : 'd', read=lambda : b'ddd'),

            b'\xc3\x9c'.decode() : mock(geturl=lambda : b'\xc3\x9c'.decode(),
                                        read=lambda : b'\xc3\x9c\xc3\x9c\xc3\x9c')}

    def mocked (cls) :
        # A mockup is made, so that the test doesn't depend on external resources (the internet).

        def open_ (url) :
            def enter (*args, **kw) :
                try :
                    return urls[url]
                except KeyError :
                    raise URLError('Something went horribly horribly wrong!!!')

            return mock(__enter__=enter, __exit__=lambda *args, **kw: None)

        class Mocked (cls) :
            def _make_opener (self) :
                return mock(open=open_)

        return Mocked

    all_ = {Response('a', 'aaa', 'a'), Response('b', 'bbb', 'b'), Response('c', 'ccc', 'c'), Response('d', 'ddd', 'd')}

    scraped = mocked(Scraped)(['a', 'b', 'c'])
    ut.assert_equal(set(scraped), {Response('a', 'aaa', 'a'), Response('b', 'bbb', 'b'), Response('c', 'ccc', 'c')})
    scraped.urls.append('d')
    ut.assert_equal(set(scraped), all_)

    # Test with generated URLs
    gen_urls = (url for url in 'abcd')
    gen_scraped = mocked(Scraped)(gen_urls)
    ut.assert_equal(set(gen_scraped), all_)
    ut.assert_equal(set(gen_scraped), set())

    cant_scrape = mocked(Scraped)(['e'], silent=False)
    ut.assert_raises(lambda : set(cant_scrape), CouldNotOpenURL)

    safe_scrape = mocked(Scraped)(['e'], silent=True)
    ut.assert_doesnt_raise(lambda : set(safe_scrape), CouldNotOpenURL)

    scraped = mocked(Scraped)(['a', 'a', 'a'], revisit=True)
    ut.assert_equal(list(scraped), [Response('a', 'aaa', 'a')] * 3)

    scraped = mocked(Scraped)(['a', 'a', 'a'], revisit=False)
    ut.assert_equal(list(scraped), [Response('a', 'aaa', 'a')])

    scraped = mocked(Scraped)([b'\xc3\x9c'.decode()], encoding='utf-8')
    # The response text should be three uppercase Us with umlauts.
    ut.assert_equal(list(scraped)[0].text, b'\xc3\x9c\xc3\x9c\xc3\x9c'.decode())

    # The scraper class should be able to handle infinite generators; if not, this test will take forever!
    @scraper(cls=mocked(Scraped), revisit=True)
    def foo () :
        while True :
            for url in ['a', 'b'] :
                yield url

    for i, response in enumerate(foo()) :
        # This shouldn't take too long to finish.
        if i == 10 :
            break

def __demo__ () :
    ''' A demonstration of the scraper decorator. '''

    @scraper(silent=True)
    def scraped () :
        yield 'http://www.wikipedia.org'
        yield 'http://python.org'
        yield 'http://missingparenthesis.com/foobar'

    for response in scraped() :
        print(repr(response)) # Notice how the URL has been switched out with a response.

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())
    __demo__()

