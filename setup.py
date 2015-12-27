from setuptools import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext

setup(
    name='hmm_kit',
    version='0.1',
    description='Python toolkit for unsupervised learning of sequences of observations using HMM',
    author='Eran Rosenthal',
    author_email='eranroz@cs.huji.ac.il',
    url='https://github.com/eranroz/hmm',
    license='MIT License',
    packages=['hmm_kit'],
    ext_modules=cythonize(['hmm_kit/_hmmc.pyx']),
    requires=['cython', 'matplotlib'],
    scripts=['scripts/simple_hmm.py'],
    build_ext=build_ext,
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research'
    ]
)
