#!/usr/bin/env python

from distutils.core import setup

setup(name='flybase_miner',
      version='0.1',
      description='A text mining app for Drosophila genes',
      author='Peter Oxley',
      author_email='oxpeter+git@gmail.com',
      url='https://github.com/oxpeter/flybase_miner',
      py_modules=['hicluster', 'common_path'],
      requires=['argparse',
                'matplotlib',
                'sklearn',
                ]
     )