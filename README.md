nlplib
======

The Natural Language Processing Library is a collection of tools for processing
natural language using Python 3. It features tools for manipulating and
extracting information from raw English language text, using multiple machine
learning and natural language processing algorithms.

## Status :
Currently it's under active development, and not feature complete. However the core library is nearly feature complete, and will undergo a feature freeze soon. There are plans to put it up on PIP, once the core reaches a reasonable level of stability. There are currently no plans to make this library
compatible with older versions of Python.

## Dependencies :
*   Python 3.3

    The library depends on new syntax introduced in Python 3.3, older versions of Python will **not** work.

*   SQLAlchemy 0.9.1

    Currently SQLAlchemy is necessary for the library to work. Though, there are are plans to implement        other storage back-ends.

#### Recommended :

*   Beautiful Soup 4.1.2

    This is used for parsing HTML collected by the web-scrapers, everything in the core will run fine           without this.

*   NumPy 1.8.0

    Everything will run fine without NumPy installed; however, having it installed will make certain neural     network algorithms run _much_ faster.

*   NLTK 3.0

    There are a few convenience functions that integrate with NLTK. However, it's not entirely necessary.

*   matplotlib 1.2.1

    This is only needed to see graphical plots of certain outputs. If it's not installed, the plots simply      won't be displayed.

## Installation :
Installation should be straight forward; it installs like pretty much every
other Python package.

```bash
$ python3 setup.py install
```
