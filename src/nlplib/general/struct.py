raise NotImplementedError("This isn't done yet.")

__all__ = ['Tree', 'Trie']

class Tree :
    __slots__ = ('parent', 'object', '_children')

    def __init__ (self, object=None, parent=None, children=()) :
        self.object = object

        self.parent = parent

        self._children = []
        self.add_children(*children)

    def __iter__ (self) :
        yield self
        for child in self._children :
            yield from child

    @staticmethod
    def objects (tree) :
        for sub_tree in tree :
            try :
                yield sub_tree.object
            except AttributeError :
                pass

    def add_children (self, *children) :
        for child in children :
            child.parent = self
        self._children.extend(children)

    def parents (self) :
        node = self
        while True :
            if node.parent is None :
                break
            else :
                yield node.parent
                node = node.parent

    def lineage (self) :
        yield self
        yield from self.parents()

class Trie :
    __slots__ = ('object', 'parent', 'children')

    def __init__ (self, object=None, parent=None) :
        self.object   = object
        self.parent   = parent
        self.children = {}

    def __iter__ (self) :
        for node in reversed(tuple(self.parents())) :
            yield node.object

    def parents (self) :
        node = self
        while True :
            if node.parent is None :
                break
            else :
                yield node
                node = node.parent

    def add (self, objects) :
        node = self
        for object in objects :
            try :
                node = node.children[object]
            except KeyError :
                node.children[object] = node = Trie(object=object, parent=node)

        return node

def kerneled (iterable, size=3) :
    iterable = tuple(iterable)

    for i, item in enumerate(iterable) :
        if (i - size) > 0 :
            before = iterable[i-size:i]
        else :
            before = iterable[:i]

        if (i + size) < len(iterable) :
            after = iterable[i+1:i+size+1]
        else :
            after = iterable[i+1:]

        yield (before, item, after)

def __test__ (ut) :
    a = Tree(object='a')
    b = Tree(object='b')
    c = Tree(object='c')
    d = Tree(object='d')
    e = Tree(object='e')
    f = Tree(object='f')
    g = Tree(object='g')

    a.add_children(b, c)
    b.add_children(d, e)
    c.add_children(f, g)

    ut.assert_equal(''.join(a.objects(a)), 'abdecfg')
    ut.assert_equal(''.join(g.objects(g.lineage())), 'gca')

    correct = [((),              'a', ('b', 'c', 'd')),
               (('a',),          'b', ('c', 'd', 'e')),
               (('a', 'b'),      'c', ('d', 'e', 'f')),
               (('a', 'b', 'c'), 'd', ('e', 'f', 'g')),
               (('b', 'c', 'd'), 'e', ('f', 'g', 'h')),
               (('c', 'd', 'e'), 'f',      ('g', 'h')),
               (('d', 'e', 'f'), 'g',          ('h',)),
               (('e', 'f', 'g'), 'h',              ())]

    for kernel, correct in zip(kerneled('abcdefgh', size=3), correct) :
        ut.assert_equal(kernel, correct)


    from pprint import pprint
    from collections import namedtuple

    prefix = Trie()
    suffix = Trie()

    Item = namedtuple('Item', ('before', 'object', 'after'))

    parsed = [Item(prefix.add(before), object, suffix.add(after)) for before, object, after in kerneled('abcdefgh')]

    item = parsed[2]

    print(list(item.before), item.object, list(item.after))

if __name__ == '__main__' :
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())

