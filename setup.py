import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="arus",
    version="0.0.1.dev0",
    author="Qu Tang",
    author_email="tang.q@husky.neu.edu",
    description="Activity Recognition with Ubiquitous Sensing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qutang/arus",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.16.3'
    ]
)