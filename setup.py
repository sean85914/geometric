import re
from setuptools import setup


with open('geometric/__init__.py') as f:
    contents = f.read()
    version = re.search(r'__version__ = [\'"]([0-9.]+)[\'"]', contents).group(1)

setup(
    name="geometric",
    version=version,
    author="SEAN.LU",
    author_email="sean19960914@gmail.com",
    description="Module for geometric calculation",
    long_description=open("README.md").read(),
    long_description_content_type='text/markdown',
    packages=["geometric", "geometric.tests"],
    install_requires=open('requirements.txt').read().splitlines(),
    python_requires='>=3.8'
)
