from setuptools import setup, find_packages

setup(
    name="paired-depth-study",
    version="0.1.0",
    packages=["tensornet"]
    + ["tensornet." + name for name in find_packages(exclude=["tests", "tests.*"])],
    package_dir={"tensornet": "."},
    python_requires=">=3.11,<3.13",
)
