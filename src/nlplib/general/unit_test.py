

import unittest
import pkgutil

__all__ = ['UnitTest', 'Mock', 'test_everything']

_log_switch = {None     : lambda output : output,
               'silent' : lambda output : output,
               'print'  : lambda output : print(output)}

def _logging_function (mode) : # This is used in <nlplib.general.time> too.
    return _log_switch.get(mode, mode)

class UnitTest :
    def __init__ (self, log='silent') :
        self._test = unittest.TestCase()
        self._logging_function = _logging_function(log)

    def log (self, output) :
        if callable(self._logging_function) :
            self._logging_function(output)
        return output

    def assert_true (self, value) :
        self.log('The value is {bool}'.format(bool=bool(value)))

        self._test.assertTrue(value)

    def assert_equal (self, value_0, value_1) :
        comparison = value_0 == value_1
        self.log('({value_0} == {value_1}) evaluates to {comparison}'.format(value_0=value_0,
                                                                             value_1=value_1,
                                                                             comparison=str(comparison).lower()))
        self._test.assertEqual(value_0, value_1)

    def assert_raises (self, function, exc) :
        self._test.assertRaises(exc, function)

class Mock :
    ''' A class for mocking objects. '''

    def __init__ (self, **attrs) :
        for name, value in attrs.items() :
            setattr(self.__class__, name, value)
            setattr(self, name, value)

def _import_everything_from (pkg) :
    for loader, name, is_pkg in pkgutil.walk_packages(pkg.__path__, onerror=lambda module : None) :
        try :
            module = loader.find_module(name).load_module(name)
        except NotImplementedError :
            pass
        else :
            yield module

def test_everything (pkg, log='print', test_function_log='silent', log_non_implemented_tests=False,
                     raise_not_implemented_error=False, test_function_name='__test__') :
    ''' This calls all module level functions named <__test__> within a package, in order to expedite package wide unit
        testing. '''

    ut = UnitTest(log)

    test_function_name = str(test_function_name)

    for module in _import_everything_from(pkg) :
        full_module_name = pkg.__name__ + '.' + module.__name__

        try :
            test_function = getattr(module, test_function_name)
        except AttributeError :
            if log_non_implemented_tests :
                ut.log('The test for module <%s> is not implemented.' % full_module_name)
        else :
            ut.log('Testing module <%s>' % full_module_name)
            try :
                test_function(UnitTest(test_function_log))
            except TypeError :
                raise TypeError('The test function named <%s> in module <%s> must accept an instance of <UnitTest> as '
                                'the first argument. Alternatively, there is a chance that the test function is '
                                'throwing TypeErrors when called.' % (test_function.__name__, full_module_name))
            except NotImplementedError :
                msg = ('Something in the function named <%s> in the module <%s> is not '
                       'implemented.' % (test_function.__name__, full_module_name))
                if raise_not_implemented_error :
                    raise NotImplementedError(msg)
                else :
                    ut.log(msg)
            else :
                ut.log('Module <%s> passed all tests!' % full_module_name)

            ut.log('') # Prints a newline

    ut.log('Done testing everything, hooray!')

def __test__ (ut) :
    mock = Mock(foo='foo',
                bar=lambda : 'bar',
                baz=lambda baz : 'baz' + baz)

    ut.assert_equal(mock.foo, 'foo')
    ut.assert_true(callable(mock.bar))
    ut.assert_equal(mock.bar(), 'bar')
    ut.assert_equal(mock.baz('baz'), 'bazbaz')

if __name__ == '__main__' :
    import nlplib
    test_everything(nlplib, log='print')

