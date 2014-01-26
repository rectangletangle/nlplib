''' Tools for dealing with multithreaded programs. '''


from concurrent.futures import ThreadPoolExecutor, as_completed

from nlplib.general.iterate import chunked

__all__ = ['simultaneously']

def simultaneously (functions, max_workers=4) :
    ''' This runs the given functions concurrently, the functions shouldn't take any arguments. The <max_workers>
        argument denotes the amount of worker threads to use. '''

    if max_workers < 1 :
        raise ValueError('<simultaneously> requires at least one worker thread.')

    with ThreadPoolExecutor(max_workers=max_workers) as executor :

        futures = (executor.submit(function)
                   for function in functions)

        for chunk in chunked(futures, max_workers, trail=True) :
            for future in as_completed(chunk) :
                yield future.result()

def __demo__ () :
    from urllib.request import urlopen

    def get_html (url) :
        return urlopen(url).read(1024)

    for html in simultaneously([lambda : get_html('http://amazon.com'),
                                lambda : get_html('http://ibm.com'),
                                lambda : get_html('http://google.com'),
                                lambda : get_html('http://python.org')]) :

        print(html, end='\n\n')

def __test__ (ut) :
    from time import sleep

    def foo () :
        sleep(0.1)
        return 'foo'

    def bar () :
        sleep(0.2)
        return 'bar'

    def baz () :
        sleep(0.3)
        return 'baz'

    ut.assert_equal(set(simultaneously([foo, bar, baz])), {'foo', 'bar', 'baz'})
    ut.assert_equal(set(simultaneously([foo, bar, baz], 1)), {'foo', 'bar', 'baz'})
    ut.assert_equal(set(simultaneously([foo, bar, baz], 231)), {'foo', 'bar', 'baz'})

    ut.assert_raises(lambda : set(simultaneously([foo, bar, baz], 0)), ValueError)
    ut.assert_raises(lambda : set(simultaneously([foo, bar, baz], -1)), ValueError)

    class SomeArbitraryException (Exception) :
        pass

    def raise_something () :
        raise SomeArbitraryException

    ut.assert_raises(lambda : list(simultaneously([raise_something])), SomeArbitraryException)

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())
    __demo__()

