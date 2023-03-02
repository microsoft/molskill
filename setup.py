from setuptools import find_packages, setup

setup(
    name="molskill",
    version="1.1",
    description="Implicit molecular scoring via chemists in the loop",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords=[],
    url="https://github.com/microsoft/molskill",
    author="Jose Jimenez-Luna",
    author_email="jjimenezluna@microsoft.com",
    license="MIT",
    packages=find_packages(),
    zip_safe=False,
)