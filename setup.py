"""
All right is reserved for UCC

"""

from setuptools import setup
from setuptools import find_packages



NAME = "odometry"
DESCRIPTION = "a library for video liveness verification based on visual odometry"
VERSION = "0.1.0"
with open("README.md", "r", encoding="utf-8") as readme:
    LONG_DESCRIPTION = readme.readlines()

AUTHOR = "Hamzeh Asadi"
LICENSE = "MIT"

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    license=LICENSE,
    packages=find_packages()
)
