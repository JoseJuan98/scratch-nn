# Contributing Guidelines

Thank you for considering contributing to this project! This guide explains how to use the tox.ini file to run tests and check for code quality before submitting a pull request.

## Installation

To use the tox.ini file, first ensure that you have tox installed. You can install tox using pip:

```
pip install --no-cache-dir .[test]
# or
pip install --no-cache-dir tox
```

Then, clone this repository to your local machine.

## Running Tests

To run tests, simply run the following command from the root directory of the repository:

```shell
tox
```

This command will create a virtual environment for each environment specified in the envlist of the tox.ini file. The test environment will run the test suite, while the flake8 and mypy environments will check for code quality.

If you are using Windows, try `python -m tox` instead of only tox.

You can also pass additional arguments to the pytest command by appending them to the tox command. For example, to run tests and show the output of print statements, run:

```shell
tox -- -s
```

### Ignoring Coverage Checks

By default, the test environment will fail if the code coverage is less than 90%. You can ignore this check by setting the IGNORE_COVERAGE environment variable to -. For example:

```bash
export IGNORE_COVERAGE=- ; python -m tox -e py310 ; unset IGNORE_COVERAGE
```

## Building and Distributing Packages

To build and check the distribution packages, use the twine environment:

```bash
tox -e twine
```

This command will build a wheel file and check the package's README for valid reStructuredText markup.

## Cleaning Up

To erase all previous coverage data, run:

```
tox -e clean
```

This will remove all previous coverage data and test results.

## Reporting

To generate a coverage report, run:

```
tox -e report
```

This will generate a coverage report based on the most recent test run.