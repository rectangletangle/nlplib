nlplib
======

The Natural language Processing Library is a collection of tools for processing
natural language using Python 3. It features tools for manipulating and
extracting information from raw English language text using several different
machine learning algorithms, and then persisting that information within a
database.

## Status :
Currently it's under active development, not feature complete, and not anywhere
near stable. There are plans to put it up on PIP, once it reaches a reasonable
level of stability. There are also plans to add a back-end for Python's
standard SQLite 3 library as well as Django's ORM, so that it doesn't fully
depend on SQLAlchemy. There are currently no plans to make this library
compatible with Python 2 or older.

## Dependencies :
* Python 3.3
* SQLAlchemy 0.8.3
* Beautiful Soup 4.1.2
* NumPy 1.7.1
* NLTK 3.0
* matplotlib 1.2.1 (only needed in order to see plots, will run without it)

## Installation :
Installation should be straight forward; it installs like pretty much every
other Python package.

```bash
$ python3 setup.py install
```
