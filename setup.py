import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="arus",
    license="GNU",
    version="0.0.1.dev4",
    author="Qu Tang",
    author_email="tang.q@husky.neu.edu",
    description="Activity Recognition with Ubiquitous Sensing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qutang/arus",
    packages=setuptools.find_packages(),
    project_urls={
        'Documentation': 'https://github.com/qutang/arus',
        'Source': 'https://github.com/qutang/arus',
        'Tracker': 'https://github.com/qutang/arus/issues',
    },
    include_package_data=True,
    install_requires=[
        'numpy>=1.16.3',
        'pandas>=0.24.2',
        'scipy>=1.3.0'
    ],
    test_suite="setup._discover_tests",
    python_requires='>= 3.5'
)
