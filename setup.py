from setuptools import setup, find_packages

setup(
    name="CLOVER",
    packages=find_packages(include=["CLOVER*"]),
    install_requires=open("requirements.txt", "r").read().splitlines(),
    python_requires=">=3.10",
    version="0.1",
)
