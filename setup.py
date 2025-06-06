from setuptools import setup


setup(
    name="geometric",
    version="0.0.0",
    author="SEAN.LU",
    author_email="sean19960914@gmail.com",
    description="Module for geometric calculation",
    long_description=open("README.md").read(),
    long_description_content_type='text/markdown',
    packages=["geometric"],
    install_requires=[
        "numpy>=1.26.2,<2.0.0",
        "scipy>=1.11.4"
    ],
    python_requires='>=3.7'
)
