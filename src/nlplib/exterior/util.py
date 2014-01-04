

import warnings

try :
    from matplotlib import pyplot
except ImportError :
    pass

from nlplib.general import iterate, math
from nlplib.general.iterate import flattened

_plot_colors = ('red', 'blue', 'green', 'magenta', 'cyan', 'yellow', 'orange', 'pink')

def _plot_basecase (lst) :
    return all(not isinstance(item, list) for item in lst)

def plot (iterable, key=lambda item : item, sample_size=1, depth=1, colors=_plot_colors, basecase=_plot_basecase,
          _show=True) :
    ''' This function can be used for plotting an iterable graphically. If matplotlib isn't installed, this will only
        throw a warning, and no GUI will be launched.. '''

    try :
        pyplot
    except NameError :
        warnings.warn('matplotlib must be installed in order to see the plot graphic.', Warning)
    else :
        for i, iterable in enumerate(flattened(iterable, basecase)) :
            xs, ys = zip(*((i, math.avg(key(item) for item in sample))
                           for i, sample in enumerate(iterate.chunked(iterable, sample_size, trail=True))))

            pyplot.plot(xs, ys, linewidth=1.0, color=colors[i % len(colors)])

        if _show :
            pyplot.show()

def __test__ (ut) :
    plot([1.0, 0.754, 0.4, 0.2, 0.01],
         _show=False)

    plot([(1.0, None), (0.754, None), (0.4, None), (0.2, None), (0.01, None)],
         lambda item : item[0],
         _show=False)

def __demo__ () :
    plot([[1.0, 0.754, 0.4, 0.2, 0.01], [1.0, 0.644, 0.24, 0.243, 0.032], [1.0, 0.93, 0.80, 0.65, 0.022]])

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest())
    __demo__()

