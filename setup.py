# Thanks:
# https://stackoverflow.com/questions/46784964/create-package-with-cython-so-users-can-install-it-without-having-cython-already
# https://stackoverflow.com/questions/2379898/make-distutils-look-for-numpy-header-files-in-the-correct-place
# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
from glob import glob
import os
from setuptools.command.build_ext import build_ext

# Checks If Cython Is Installed
try:
    from Cython.Build import cythonize
except Exception as e:
    print(e)
    # If we couldn't import Cython, use the normal setuptools
    # and look for a pre-compiled .c file instead of a .pyx file
    USING_CYTHON = False
else:
    # If we successfully imported Cython, look for a .pyx file
    USING_CYTHON = True

from distutils.core import setup
from distutils.extension import Extension

class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""
    def run(self):

        # Import numpy here, only when headers are needed
        import numpy

        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())

        # Call original build_ext command
        build_ext.run(self)

# Get All Extension Modules
ext = 'pyx' if USING_CYTHON else 'c'
sources = glob('knn/*.%s' % (ext,))
ext_modules = [
    Extension(source.split('.')[0].replace(os.path.sep, '.'),
              sources=[source],)
    for source in sources]

setup(
    name='knn',
    version='0.1',
    description='KNN using brute force and ball trees implemented in Python/Cython',
    url='https://github.com/JKnighten/k-nearest-neighbors',
    author='Jonathan Knighten',
    author_email='jknigh28@gmail.com',
    cmdclass={'build_ext': CustomBuildExtCommand},
    install_requires=['numpy'],
    ext_modules=cythonize(ext_modules, language_level=3, annotate=True),
    packages=['knn'],
    zip_safe=False
)
