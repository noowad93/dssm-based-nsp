from setuptools import find_packages, setup

setup(
    name="dssm-based-nsp",
    version="0.0.1",
    description="Deep Structured Semantic Models-based Next Sentence Prediction",
    install_requires=[],
    url="https://github.com/noowad93/dssm-based-nsp.git",
    author="ScatterLab",
    author_email="developers@scatterlab.co.kr",
    packages=find_packages(exclude=["tests"]),
)
