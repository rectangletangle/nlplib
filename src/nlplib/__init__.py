

''' nlplib development version '''

__version__      = '0.0.0'
__author__       = 'Drew A. French'
__email__        = 'rectangletangle@gmail.com'
__url__          = 'missingparenthesis.com'
__download_url__ = 'github.com/rectangletangle/nlplib'
__licence__      = 'Simplified BSD License'

try :

    from nlplib.general.iterate import windowed, chunked, paired, flattened
    from nlplib.general.scrape import Scraped, scraper, CouldNotOpenURL
    from nlplib.general.thread import simultaneously

    from nlplib.core.model import Database, Seq, Word, Gram, Document, NeuralNetwork
    from nlplib.core.model.exc import StorageError
    from nlplib.core.process.index import Indexed
    from nlplib.core.process.token import re_tokenized, split_tokenized, split
    from nlplib.core.process.parse import Parsed
    from nlplib.core.exc import NLPLibError
    from nlplib.core.control.score import Score, Scored, ScoredAgainst
    from nlplib.core.control.neuralnetwork.structure import StaticLayer, StaticIOLayer

    try :
        from nlplib.core.process.token import nltk_tokenized
    except ImportError :
        ...

    from nlplib.data import builtin_db

except ImportError :
    # This occurs when setup.py imports version information, just before nlplib is installed.
    ...

