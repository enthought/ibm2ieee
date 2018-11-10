from setuptools import Extension, find_packages, setup

import numpy


ibm2ieee_extension = Extension(
    name="_ibm2ieee",
    sources=[
        "ibm2ieee/_ibm2ieee.c",
    ],
    include_dirs=[numpy.get_include()],
)


if __name__ == "__main__":
    setup(
        packages=find_packages(),
        ext_modules=[ibm2ieee_extension],
    )
