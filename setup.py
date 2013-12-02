

import sys

from version_check import version_check, import_check, version_as_string

def check_for_nlplibs_dependencies (oldest_python_version_tolerated, oldest_sqlalchemy_version_tolerated,
                                    oldest_bs_version_tolerated) :
    too_old_template = 'This package requires {name} {version} or greater.'

    version_check(version_as_string(sys.version_info[:3]),
                  oldest_python_version_tolerated,
                  too_old_template.format(name='Python', version=oldest_python_version_tolerated))

    import_check('sqlalchemy',
                 oldest_sqlalchemy_version_tolerated,
                 too_old_template.format(name='SQLAlchemy', version=oldest_sqlalchemy_version_tolerated))

    import_check('bs4',
                 oldest_bs_version_tolerated,
                 too_old_template.format(name='Beautiful Soup', version=oldest_bs_version_tolerated))

check_for_nlplibs_dependencies(oldest_python_version_tolerated='3.3',
                               oldest_sqlalchemy_version_tolerated='0.8.3',
                               oldest_bs_version_tolerated='4.1.2')

from distutils.core import setup

from src import nlplib

packages = ['nlplib',
            'nlplib.core',
            'nlplib.core.acquire',
            'nlplib.core.acquire.scrape',
            'nlplib.core.model',
            'nlplib.core.model.backend',
            'nlplib.core.model.backend.abstract',
            'nlplib.core.model.backend.sqlite3',
            'nlplib.core.model.backend.sqlalchemy',
            'nlplib.core.model.backend.sqlalchemy.map',
            'nlplib.core.model.backend.django',
            'nlplib.core.process',
            'nlplib.core.score',
            'nlplib.core.score.metric',
            'nlplib.core.control',
            'nlplib.core.control.neural_network',
            'nlplib.ext',
            'nlplib.general',
            'nlplib.data',
            'nlplib.scripts']

setup(name = 'nlplib',

      version      = nlplib.__version__,
      author       = nlplib.__author__,
      author_email = nlplib.__email__,
      url          = nlplib.__url__,
      download_url = nlplib.__download_url__,
      license      = nlplib.__licence__,

      description      = '',
      long_description = nlplib.__doc__.strip(),

      packages     = packages,
      package_dir  = {'nlplib' : 'src/nlplib'},
      package_data = {'nlplib' : ['data/builtin.db']},

      requires = ['beautifulsoup (>=4.1.2)', 'sqlalchemy (>=0.8.3)'])

