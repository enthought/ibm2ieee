# Copyright (c) 2018, Enthought, Inc.
# All rights reserved.
from __future__ import absolute_import, print_function

import io
import os

import numpy
import setuptools


def get_version_info():
    """ Extract version information as a dictionary from version.py. """
    version_info = {}
    version_filename = os.path.join("ibm2ieee", "version.py")
    with io.open(version_filename, "r", encoding="utf-8") as version_module:
        version_code = compile(version_module.read(), "version.py", "exec")
        exec(version_code, version_info)
    return version_info


def get_long_description():
    """ Read long description from README.txt. """
    with io.open("README.rst", "r", encoding="utf-8") as readme:
        return readme.read()


ibm2ieee_extension = setuptools.Extension(
    name="ibm2ieee._ibm2ieee",
    sources=[
        "ibm2ieee/_ibm2ieee.c",
    ],
    define_macros=[
        ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
    ],
    include_dirs=[numpy.get_include()],
)

SHORT_DESCRIPTION = u"""\
Convert IBM hexadecimal floating-point data to IEEE 754 floating-point data.
""".rstrip()


if __name__ == "__main__":
    setuptools.setup(
        name="ibm2ieee",
        version=get_version_info()["version"],
        author="Enthought",
        author_email="info@enthought.com",
        url="https://github.com/enthought/ibm2ieee",
        description=SHORT_DESCRIPTION,
        long_description=get_long_description(),
        long_description_content_type="text/x-rst",
        keywords="ibm hfp ieee754 hexadecimal floating-point ufunc",
        install_requires=["numpy"],
        extras_require={
            "test": ["packaging"],
        },
        packages=setuptools.find_packages(),
        ext_modules=[ibm2ieee_extension],
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: BSD License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
        ],
    )
