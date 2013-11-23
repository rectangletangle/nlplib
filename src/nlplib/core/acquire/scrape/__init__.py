

from urllib.request import build_opener, URLError

try :
    from bs4 import BeautifulSoup
except ImportError :
    raise ImportError(r'This package requires Beautiful Soup, http://www.crummy.com/software/BeautifulSoup/')

from nlplib.general.threaded import simultaneously
from nlplib.general.iter import chunked
from nlplib.core import Base

__all__ = ['WebScraper', 'scrape_simultaneously']

def _add_more_info_to_exception (exception, info) :
    try :
        exception.args += (info,)
    except Exception :
        pass

    return exception

class WebScraper (Base) :
    ''' A base web scraper class. '''

    _user_agent = 'Mozilla/5.0'

    def __init__ (self) :
        self._opener = build_opener()
        self._opener.addheaders[:] = (('User-agent', self._user_agent),)

        self._visited_urls = set()

    def _open (self, url) :
        return self._opener.open(url)

    def _parse (self, html) :
        return BeautifulSoup(html)

    def _the_url_has_been_visited_before (self, url) :
        if url in self._visited_urls :
            return True
        else :
            self._visited_urls.add(url)
            return False

    def _been_at_url_before (self, function, url, page) :
        pass

    def _not_been_at_url_before (self, function, url, page) :
        html = page.read()
        soup = self._parse(html)

        function(url, soup)

    def _could_open_url (self, function, url, page) :
        current_url = page.geturl()

        if self._the_url_has_been_visited_before(current_url) :
            self._been_at_url_before(function, current_url, page)
        else :
            self._not_been_at_url_before(function, current_url, page)

    def _could_not_open_url (self, function, url, exception) :
        raise _add_more_info_to_exception(exception, 'happened with the url : %s' % str(url))

    def scrape (self, function, url) :
        try :
            with self._open(url) as page :
                self._could_open_url(function, url, page)
        except Exception as exception :
            self._could_not_open_url(function, url, exception)

def scrape_simultaneously (scraping_functions, at_a_time=20) :
    for broken_up_functions in chunked(scraping_functions, at_a_time) :
        simultaneously(broken_up_functions)

