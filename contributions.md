# Contributing to the project

Welcome! This guide explains the internal structure and technical details of this codebase for high-resolution video registration of concrete samples.

## Project Overview

See the [README.md](/README.md) for the basic introduction.

The project has two main directories:

1. `src`: This is where the whole software resides, together with its modules.
2. `tests`: Contains `PyTest` tests that are used for ensuring that adding new feature don't (completely) break the existing functionality.

Futhermore, there are the following files:

1. [`pyproject.toml`](/pyproject.toml): Used for installing the cli program with `Poetry`
2. [`README.md`](/README.md): Simple introduction to the project and how to get it running.
3. [`future_work.md`](/future_work.md): A short list of ideas that could be used for future work on the project.

# Tests

The tests that have been written cover the program's functionality so far. They don't use real data, instead we created mockups that have the same "signature" as our data.

`PyTest` is used for running these tests. If you want to only run a certain test, you can do so with the following `pytest tests/test_filename.py`. 

These tests definitely **do not** cover all possible edge-cases, nor do they cover "invalid" filetypes. As this tool is being mainly developed for only internal uses, the program expects that the data that is passed into it has the valid type and that it contains what we would expect it to contain (i.e. a concrete sample that has been radioactively exposed).

# Src

The software consists of several modules that can be either be used in a pipeline manner, or independently. The modules that have been developed so far are:

1. [`crack identification`](/src/concrete_samples_toolkit/crack_identification):
1. [``]()
