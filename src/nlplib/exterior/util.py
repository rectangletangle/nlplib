

from matplotlib import pyplot

from nlplib.general import iterate, math

def plot_errors (trained, chunk_size=1) :
    xs = []
    ys = []
    for i, chunk in enumerate(iterate.chunked(trained, chunk_size)) :
        xs.append(i)
        ys.append(math.avg(error for error, *other in chunk))

    pyplot.plot(xs, ys, linewidth=1.0)
    pyplot.show()

def __demo__ () :
    trained = [(1.0, None), (0.754, None), (0.4, None), (0.2, None), (0.01, None)]
    plot_errors(trained)

if __name__ == '__main__' :
    __demo__()

