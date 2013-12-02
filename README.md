nlplib
======

The Natural language Processing Library (nlplib) is a library of tools for
processing natural language using Python 3. It features tools for manipulating
and extracting information from raw English language text, and then persisting
that information within a database.

# Status :
Currently it's under active development, not feature complete, and not anywhere
near stable. There are plans to put it up on PIP, once it reaches a reasonable
level of stability. There are also plans to add a back-end for Python's
standard SQLite 3 library as well as Django's ORM, so that it doesn't fully
depend on SQLAlchemy. There are currently no plans to make this library
compatible with Python 2 or older.

# Dependencies :
* Python 3.3
* SQLAlchemy 0.8.3
* Beautiful Soup 4.1.2

# Installation :
Installation should be straight forward; it installs like pretty much every
other Python package.

```
$ python3 setup.py install
```
