
import os
from setuptools import setup, find_namespace_packages, find_packages

with open(os.path.join(os.path.dirname(__file__), "README.md"), "r") as fh:
    long_description = fh.read()

with open(os.path.join(os.path.dirname(__file__), "requirements.txt"), 'r') as fh:
    dependencies = fh.readlines()
    dependencies = [d.strip() for d in dependencies]

setup(
    name="arus",
    license="GNU",
    version="0.0.2",
    author="Qu Tang",
    author_email="tang.q@husky.neu.edu",
    description="Activity Recognition with Ubiquitous Sensing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qutang/arus",
    packages=find_packages(),
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
