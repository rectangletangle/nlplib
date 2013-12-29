

import warnings

try :
    from matplotlib import pyplot
except ImportError :
    pass

from nlplib.general import iterate, math

def plot (iterable, key=lambda item : item, chunk_size=1, show=True) :
    try :
        pyplot
    except NameError :
        warnings.warn('matplotlib must be installed in order to see the plot graphic.', Warning)
    else :
        xs, ys = zip(*((i, math.avg(key(item) for item in chunk))
                       for i, chunk in enumerate(iterate.chunked(iterable, chunk_size))))

        pyplot.plot(xs, ys, linewidth=1.0)

        if show :
            pyplot.show()

def __test__ (ut) :
    plot([1.0, 0.754, 0.4, 0.2, 0.01],
         show=False)

    plot([(1.0, None), (0.754, None), (0.4, None), (0.2, None), (0.01, None)],
         lambda item : item[0],
         show=False)

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())

