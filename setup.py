from setuptools import setup, find_packages

setup(
    name="tunix-dpo",
    version="0.2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
