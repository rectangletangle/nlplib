

import random

from nlplib.general.represent import pretty_ellipsis
from nlplib.core.base import Base

__all__ = ['random_weights', 'LayerConfiguration', 'Static', 'StaticIO', 'Dynamic', 'DynamicIO']

def random_weights (floor=-1.0, ceiling=1.0) :
    ''' This can be used to initialize the connections in the neural network with pseudorandom weights. '''

    while True :
        yield random.uniform(floor, ceiling)

class LayerConfiguration (Base) :
    pass

class Static (LayerConfiguration) :
    def __init__ (self, size) :
        self.size = size

    def __repr__ (self, *args, **kw) :
        return super().__repr__(self.size, *args, **kw)

    def __iter__ (self) :
        for _ in range(self.size) :
            yield None

    def __len__ (self) :
        return self.size

class StaticIO (LayerConfiguration) :
    def __init__ (self, objects) :
        self.objects = list(objects)

    def __repr__ (self, *args, **kw) :
        limit = 5

        if len(self.objects) > limit :
            pretty_objects = tuple(self.objects[:limit] + [pretty_ellipsis()])
        else :
            pretty_objects = tuple(self.objects)

        return super().__repr__(*pretty_objects + args, **kw)

    def __iter__ (self) :
        return iter(self.objects)

    def __len__ (self) :
        return len(self.objects)

class Dynamic (LayerConfiguration) :
    def __init__ (self, *args, **kw) :
        raise NotImplementedError

class DynamicIO (LayerConfiguration) :
    def __init__ (self, *args, **kw) :
        raise NotImplementedError

