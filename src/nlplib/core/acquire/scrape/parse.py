

try :
    from bs4 import BeautifulSoup
except ImportError :
    raise ImportError(r'This package requires Beautiful Soup, http://www.crummy.com/software/BeautifulSoup/')

__all__ = ['parse_html']

def parse_html (html) :
    ''' This parses HTML into a beautiful soup. '''

    return BeautifulSoup(html)

