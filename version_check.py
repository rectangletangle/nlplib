''' This is a tiny library for dependency and version checking. '''


import importlib

def version_as_tuple (version_string) :
    return tuple(int(value) for value in version_string.split('.'))

def version_as_string (version_tuple) :
    return '.'.join(str(value) for value in version_tuple)

def version_check (version, oldest_version_tolerated_string, msg) :
    if version_as_tuple(version) < version_as_tuple(oldest_version_tolerated_string) :
        raise Exception(msg)

def import_check (module_name, oldest_version_tolerated_string, msg) :
    try :
        module = importlib.import_module(module_name)
    except ImportError :
        raise ImportError(msg)

    version_check(module.__version__, oldest_version_tolerated_string, msg)

