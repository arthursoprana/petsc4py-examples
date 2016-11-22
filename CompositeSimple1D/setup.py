#!/usr/bin/env python

#$ python setup.py build_ext --inplace

from numpy.distutils.command import build_src

# a bit of monkeypatching ...
import Cython.Compiler.Main
build_src.Pyrex = Cython
build_src.have_pyrex = lambda: True

def find_petsc_dir():
    import petsc4py
    petsc4py.init()
    path, arch = petsc4py.lib.getPathArchPETSc()
    filename = path + '\\petsc.cfg'
    f = open(filename, 'r')

    petsc_dir = f.readline()[:-1]
    petsc_arch = f.readline()
    var_name, petsc_dir = [x.strip() for x in petsc_dir.split('=')]
    return petsc_dir
    
def have_pyrex():
    import sys
    try:
        import Cython.Compiler.Main
        sys.modules['Pyrex'] = Cython
        sys.modules['Pyrex.Compiler'] = Cython.Compiler
        sys.modules['Pyrex.Compiler.Main'] = Cython.Compiler.Main
        return True
    except ImportError:
        return False
build_src.have_pyrex = have_pyrex

##########################
# BEGIN additionnal code #
##########################
from numpy.distutils.misc_util import appendpath
from numpy.distutils import log
from os.path import join as pjoin, dirname
from distutils.dep_util import newer_group
from distutils.errors import DistutilsError


def generate_a_pyrex_source(self, base, ext_name, source, extension):
    ''' Monkey patch for numpy build_src.build_src method
    Uses Cython instead of Pyrex.
    Assumes Cython is present
    '''
    if self.inplace:
        target_dir = dirname(base)
    else:
        target_dir = appendpath(self.build_src, dirname(base))
    target_file = pjoin(target_dir, ext_name + '.c')
    depends = [source] + extension.depends
    if self.force or newer_group(depends, target_file, 'newer'):
        import Cython.Compiler.Main
        log.info("cythonc:> %s" % (target_file))
        self.mkpath(target_dir)
        options = Cython.Compiler.Main.CompilationOptions(
            defaults=Cython.Compiler.Main.default_options,
            include_path=extension.include_dirs,
            output_file=target_file)
        cython_result = Cython.Compiler.Main.compile(source, options=options)
        if cython_result.num_errors != 0:
            raise DistutilsError("%d errors while compiling %r with Cython" % (cython_result.num_errors, source))
    return target_file

build_src.build_src.generate_a_pyrex_source = generate_a_pyrex_source
########################
# END additionnal code #
########################


def configuration(parent_package='',top_path=None):
    INCLUDE_DIRS = []
    LIBRARY_DIRS = []
    LIBRARIES    = []

    # PETSc
    import os
    PETSC_DIR = find_petsc_dir()    
    PETSC_ARCH = os.environ.get('PETSC_ARCH', '')
    
    from os.path import join, isdir
    if PETSC_ARCH and isdir(join(PETSC_DIR, PETSC_ARCH)):
        INCLUDE_DIRS += [join(PETSC_DIR, PETSC_ARCH, 'include'),
                         join(PETSC_DIR, 'include')]
        LIBRARY_DIRS += [join(PETSC_DIR, PETSC_ARCH, 'lib')]
    else:
        if PETSC_ARCH: pass # XXX should warn ...
        INCLUDE_DIRS += [join(PETSC_DIR, 'include'), join(PETSC_DIR, 'include\\petsc\\mpiuni')]
        LIBRARY_DIRS += [join(PETSC_DIR, 'lib')]
    LIBRARIES += ['petsc']

    # PETSc for Python
    import petsc4py
    INCLUDE_DIRS += [petsc4py.get_include()]

    # Configuration
    from numpy.distutils.misc_util import Configuration
    config = Configuration('', parent_package, top_path)
    config.add_extension('CompositeSimple1D',
                         sources = ['CompositeSimple1D.pyx',
                                    'CompositeSimple1Dimpl.c'],
                         depends = ['CompositeSimple1Dimpl.h'],
                         include_dirs=INCLUDE_DIRS + [os.curdir],
                         libraries=LIBRARIES,
                         library_dirs=LIBRARY_DIRS,
                         extra_compile_args=["-Zi", "/Od"],
                         extra_link_args=["-debug"],
                         )
                         #runtime_library_dirs=LIBRARY_DIRS)

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
