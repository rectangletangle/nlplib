

from concurrent.futures import ThreadPoolExecutor

__all__ = ['simultaneously']

def simultaneously (functions, max_workers=None) :
    ''' This runs the given functions concurrently. The functions shouldn't take any arguments, and any return values
        will be ignored, for proper usage look at this module's demo function <__demo__>. '''

    functions = list(functions)
    max_workers = max_workers if max_workers is not None else len(functions)

    with ThreadPoolExecutor(max_workers=max_workers) as executor :
        futures = [executor.submit(function) for function in functions]

        for future in futures :
            while future.running() :
                pass
            future.result()

def __demo__ () :
    from urllib.request import urlopen

    def get_html (url) :
        return urlopen(url).read(1024)

    # Because the functions' return values are ignored, we append them to a list outside of the functions scope.
    storage = []

    simultaneously([lambda : storage.append(get_html('http://amazon.com')),
                    lambda : storage.append(get_html('http://ibm.com')),
                    lambda : storage.append(get_html('http://google.com')),
                    lambda : storage.append(get_html('http://python.org'))])

    for html in storage :
        print(html, end='\n\n')

def __test__ (ut) :
    from time import sleep

    storage = set()

    def foo () :
        sleep(0.1)
        storage.add('foo')

    def bar () :
        sleep(0.2)
        storage.add('bar')

    def baz () :
        sleep(0.3)
        storage.add('baz')

    simultaneously([foo, bar, baz])

    ut.assert_equal(storage, {'foo', 'bar', 'baz'})

    class FooError (Exception) :
        pass

    def raise_foo () :
        raise FooError

    ut.assert_raises(lambda : simultaneously([raise_foo]), FooError)

if __name__ == '__main__' :
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())
    __demo__()

