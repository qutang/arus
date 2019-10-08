import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as fh:
    dependencies = fh.readlines()
    dependencies = [d.strip() for d in dependencies]

setuptools.setup(
    name="arus",
    license="GNU",
    version="0.0.1.dev12",
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
    install_requires=dependencies,
    test_suite="setup._discover_tests",
    python_requires='>= 3.5'
)
