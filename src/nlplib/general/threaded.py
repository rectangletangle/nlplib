

from threading import Thread
from queue import Queue

__all__ = ['simultaneously']

class _Simultaneously (Thread) :
    def __init__(self, queue) :
        Thread.__init__(self)
        self.queue = queue
        self.daemon = True
        self.start()

    def run (self) :
        function = self.queue.get()
        try :
            function()
        except TypeError :
            pass
        self.queue.task_done()

def simultaneously (functions) :
    ''' This runs the given functions concurrently. '''

    function_queue = Queue()

    for function in functions :
        function_queue.put(function)
        _Simultaneously(function_queue)

    function_queue.join()

def __demo__ () :
    from urllib.request import urlopen

    storage = []

    def get_html (url) :
        storage.append(urlopen(url).read(1024))

    simultaneously([lambda : get_html('http://amazon.com'),
                    lambda : get_html('http://ibm.com'),
                    lambda : get_html('http://google.com')])

    for html in storage :
        print(html, end='\n\n')

def __test__ (ut) :
    from threading import active_count
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
    ut.assert_equal(active_count(), 1)

if __name__ == '__main__' :
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())
    __demo__()

