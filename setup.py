"""
All right is reserved for UCC

"""

from setuptools import setup
from setuptools import find_packages



NAME = "odometry"
DESCRIPTION = "a library for video liveness verification based on visual odometry"
VERSION = "0.1.0"
LONG_DESCRIPTION = ""
with open("README.md", "r", encoding="utf-8") as readme:
    lines = readme.readlines()
    for line in lines:
        LONG_DESCRIPTION += line

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
