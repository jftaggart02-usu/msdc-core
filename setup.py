import os

from setuptools import setup, find_packages

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'

install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setup(
    name="msdc_core",
    install_requires=install_requires,
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])
)
