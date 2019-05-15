.PHONY: all

all: packaging upload

packaging setup.py:
	python3 setup.py sdist bdist_wheel

upload dist/*:
	python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*