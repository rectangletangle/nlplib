''' Tools for scraping text from the internet. '''


import functools

from urllib.request import build_opener, URLError

from nlplib.general.represent import pretty_truncate, pretty_string
from nlplib.general.thread import simultaneously
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
        return super().__repr__(pretty_string(self.url), text=pretty_truncate(self.text), *args, **kw)

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
    ''' The web scraper class.

        urls : An iterable of URL strings; this can be an infinite generator.

        silent : If this is true, the scraper will ignore URLs that it can't open/handle.

        max_workers : The amount of worker threads to use when scraping concurrently, if the <serial> argument is true
        this will be ineffectual.

        revisit : If true, the scraper will revisit duplicate URLs.

        serial : This makes the scraper scrape from the URLs serially, as opposed to concurrently.

        user_agent : The value for the user-agent HTTP header.

        encoding : The text encoding scheme. '''

    def __init__ (self, urls, silent=False, max_workers=10, revisit=True, serial=False, user_agent='nlplib',
                  encoding='utf-8') :

        self.urls = urls

        self.silent = silent
        self.revisit = revisit

        self.max_workers = max_workers
        self.serial = serial

        self.user_agent = user_agent
        self.encoding = encoding

        self._visited_urls = set()

    def __iter__ (self) :
        self._visited_urls.clear()

        opener = self._make_opener()

        if self.serial :
            responses = (self._scrape(opener, url) for url in self.urls)
        else :
            responses = simultaneously(lambda url : self._scrape(opener, url), self.urls, max_workers=self.max_workers)

        for response in responses :
            if response is not None :
                yield response

    def not_been_at_url_before (self, response) :
        ''' The behavior for when the scraper gets a response from a URL it hasn't seen before. '''

        return response

    def been_at_url_before (self, response) :
        ''' Behavior for when the scraper gets a response from a URL it has seen before. '''

        pass

    def could_not_open_url (self, exc, url) :
        ''' Behavior for when the scraper couldn't get a response from the url. If the <silent> attribute is
            false (the default), this will throw exceptions. '''

        if not self.silent :
            raise CouldNotOpenURL("The URL <{}> couldn't be opened.".format(str(url))) from exc

    def read (self, page) :
        ''' Behavior for when a page is being read into memory, you probably only want to override this when dealing
            with huge pages. '''

        return page.read().decode(self.encoding)

    def _make_opener (self) :
        opener = build_opener()
        opener.addheaders[:] = (('User-agent', self.user_agent),)
        return opener

    def _scrape (self, opener, request_url) :
        try :
            with opener.open(request_url) as page :
                response = Response(url=page.geturl(), text=self.read(page), request_url=request_url)
                return self._could_open_url(response)
        except (UnicodeError, MemoryError, URLError) as exc :
            return self.could_not_open_url(exc, request_url)

    def _could_open_url (self, response) :
        if response.url in self._visited_urls :
            return self.been_at_url_before(response)
        else :
            if not self.revisit :
                self._visited_urls.add(response.url)

            return self.not_been_at_url_before(response)

def scraper (generator=None, *args, cls=Scraped, **kw) :
    ''' This decorator takes a generator function that yields URLs, and makes it yield the responses from those URLs.
        The <Scraped> class has been specifically designed to allow for the use of infinite generator functions. '''

    def wrapper (generator) :
        @functools.wraps(generator)
        def with_generator (*wrapped_args, **wrapped_kw) :
            return cls(generator(*wrapped_args, **wrapped_kw), *args, **kw)
        return with_generator

    if callable(generator) :
        # This allows the decorator to be used without arguments.
        # @scraper
        # def foo () :
        #     ...

        return wrapper(generator)
    else :
        return wrapper

def _test_with_memory_errors (ut, mock, mocked, **kw) :

    def memory_error_while_reading () :
        raise MemoryError('This is huge!')

    urls = {'a' : mock(geturl=lambda : 'a', read=memory_error_while_reading)}

    ut.assert_raises(lambda : list(mocked(Scraped, urls=urls)(['a'], **kw)), CouldNotOpenURL)
    ut.assert_doesnt_raise(lambda : list(mocked(Scraped, urls=urls)(['a'], silent=True, **kw)), CouldNotOpenURL)

def _test_with_unicode (ut, mocked, **kw) :
    ''' The response text should be three uppercase Us with umlauts. '''

    scraped = mocked(Scraped)([b'\xc3\x9c'.decode()], encoding='utf-8', **kw)
    ut.assert_equal(list(scraped)[0].text, b'\xc3\x9c\xc3\x9c\xc3\x9c'.decode())

def _test_revisit (ut, mocked, **kw) :
    ''' Tests the behavior for the <revisit> argument. '''

    scraped = mocked(Scraped)(['a', 'a', 'a'], revisit=True, **kw)
    ut.assert_equal(list(scraped), [Response('a', 'aaa', 'a')] * 3)

    scraped = mocked(Scraped)(['a', 'a', 'a'], revisit=False, **kw)
    ut.assert_equal(list(scraped), [Response('a', 'aaa', 'a')])

def _test_when_unavailable (ut, mocked, **kw) :
    ''' Tests the behavior when an unavailable resource is encountered. '''

    cant_scrape = mocked(Scraped)(['e'], silent=False, **kw)
    ut.assert_raises(lambda : set(cant_scrape), CouldNotOpenURL)

    safe_scrape = mocked(Scraped)(['e'], silent=True, **kw)
    ut.assert_doesnt_raise(lambda : set(safe_scrape), CouldNotOpenURL)

def _test_with_infinite_generator (mocked, **kw) :
    ''' The scraper class should be able to handle infinite generators; if not, this test will take forever! A real
        unit test can't be done here, because it would basically involve solving the halting problem. '''

    @scraper(cls=mocked(Scraped), revisit=True, **kw)
    def foo () :
        while True :
            for url in ['a', 'b'] :
                yield url

        for i, response in enumerate(foo()) :
            # This shouldn't take too long to finish.
            if i == 10 :
                break

def __test__ (ut) :
    from nlplib.general.unittest import mock

    urls = {'a' : mock(geturl=lambda : 'a', read=lambda : b'aaa'),
            'b' : mock(geturl=lambda : 'b', read=lambda : b'bbb'),
            'c' : mock(geturl=lambda : 'cc', read=lambda : b'ccc'),
            'd' : mock(geturl=lambda : 'd', read=lambda : b'ddd'),

            b'\xc3\x9c'.decode() : mock(geturl=lambda : b'\xc3\x9c'.decode(),
                                        read=lambda : b'\xc3\x9c\xc3\x9c\xc3\x9c')}

    def mocked (cls, urls=urls) :
        # A mockup is made, so that this test doesn't depend on external resources (the internet).

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

    all_ = {Response('a', 'aaa', 'a'), Response('b', 'bbb', 'b'), Response('cc', 'ccc', 'c'),
            Response('d', 'ddd', 'd')}

    for kw in [{'serial' : False}, {'serial' : True}] :
        scraped = mocked(Scraped)(['a', 'b', 'c'], **kw)

        ut.assert_equal(set(scraped),
                        {Response('a', 'aaa', 'a'), Response('b', 'bbb', 'b'), Response('cc', 'ccc', 'c')})

        scraped.urls.append('d')
        ut.assert_equal(set(scraped), all_)

        # Test with generated URLs
        gen_urls = (url for url in 'abcd')
        gen_scraped = mocked(Scraped)(gen_urls, **kw)
        ut.assert_equal(set(gen_scraped), all_)
        ut.assert_equal(set(gen_scraped), set())

        _test_with_memory_errors(ut, mock, mocked, **kw)
        _test_with_unicode(ut, mocked, **kw)
        _test_revisit(ut, mocked, **kw)
        _test_when_unavailable(ut, mocked, **kw)
        _test_with_infinite_generator(mocked, **kw)

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

