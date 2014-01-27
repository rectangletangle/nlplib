''' Tools for dealing with multithreaded programs. '''


from concurrent.futures import ThreadPoolExecutor, as_completed

from nlplib.general.iterate import chunked

__all__ = ['simultaneously']

def simultaneously (function, iterable, max_workers=4) :
    ''' This runs the given function over the iterable concurrently, in a similar fashion to the built-in <map>
        function. The output's order is not guaranteed to correspond the order of the input iterable. Therefor the
        output order should be treated as undefined. The <max_workers> argument denotes the amount of worker threads to
        use. '''

    if max_workers < 1 :
        raise ValueError('<simultaneously> requires at least one worker thread.')

    with ThreadPoolExecutor(max_workers=max_workers) as executor :

        futures = (executor.submit(function, item)
                   for item in iterable)

        for chunk in chunked(futures, max_workers, trail=True) :
            for future in as_completed(chunk) :
                yield future.result()

def __demo__ () :
    from urllib.request import urlopen

    urls = ['http://amazon.com', 'http://ibm.com', 'http://google.com', 'http://python.org']

    for html in simultaneously(lambda url : urlopen(url).read(1024), urls) :

        print(html, end='\n\n')

def __test__ (ut) :

    def double (string) :
        return string * 2

    inputs = ['foo', 'bar', 'baz']
    outputs = {'foofoo', 'barbar', 'bazbaz'}

    for kw in [{}, {'max_workers' : 1}, {'max_workers' : 231}] :
        ut.assert_equal(set(simultaneously(double, inputs, **kw)), outputs)

    for workers in [0, -1, -13421] :
        ut.assert_raises(lambda : set(simultaneously(double, inputs, max_workers=workers)), ValueError)

    class SomeArbitraryException (Exception) :
        pass

    def raise_something (string) :
        raise SomeArbitraryException

    ut.assert_raises(lambda : list(simultaneously(raise_something, inputs)), SomeArbitraryException)

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())
    __demo__()

