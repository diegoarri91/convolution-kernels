import os
from setuptools import setup
from typing import List


LONG_DESCRIPTION = """Python package that implements kernels/filters convolutions for time series modeling. Kernels 
                      are defined as different linear combinations of basis functions with coefficients. Convolutions
                      are implemented using PyTorch so we can use its automatic differentiation to fit the kernels."""
PATH_ROOT = os.path.dirname(__file__)

def _load_requirements(path_dir: str , file_name: str = 'requirements.txt', comment_char: str = '#') -> List[str]:
    """Load requirements from a file
    >>> _load_requirements(PATH_ROOT)
    ['numpy...', 'torch...', ...]
    """
    with open(os.path.join(path_dir, file_name), 'r') as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[:ln.index(comment_char)].strip()
        if ln:  # if requirement is not empty
            reqs.append(ln)
    print(reqs)
    return reqs

setup(
    name='convolution-kernels',
    version='0.1',
    description='Python package that implements kernels/filters convolutions for time series modeling.',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/diegoarri91/convolution-kernels',
    author='Diego M. Arribas',
    author_email='diegoarri91@gmail.com',
    license='MIT',
    classifiers=[
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
    ],
    keywords=['python', 'pytorch', 'convolution', 'kernel', 'autograd'],
    packages=['kernel'],
    include_package_data=True,
    install_requires=_load_requirements(PATH_ROOT),
    python_requires='>=3',
)
