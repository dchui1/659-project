from distutils.core import setup, Extension
import numpy as np

module1 = Extension('sparse',
                    sources = ['sparse_module.c'])

setup (name = 'sparse',
       version = '1.0',
       description = 'This is a demo package',
       ext_modules = [module1],
       include_dirs = [np.get_include()])

# setup (name = 'tiles',
#        version = '1.0',
#        description = 'This is a demo package',
#        ext_modules = [module2],
#        include_dirs = [np.get_include()])
