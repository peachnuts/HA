# HA algorithm

This repository contains the implementation of our Hardware-Aware mapping algorithm.

## Installation

The repository is a Python package. You can install by cloning with `git` and then using Python's package manager `pip`:

``` shell
git clone https://github.com/peachnuts/HA.git
python -m pip install HA
```

## Notes on the implementation

The implementation presented here uses a slightly different method to chose between inserting a `SWAP` or a `Bridge` gate.
The algorithm described in the scientific paper first computes the best `SWAP` and then determine if it is worth changing the `SWAP` into a `Bridge` gate.
The implementation in this repository evaluates `Bridge` gates along `SWAP` ones, and pick the best gate according to the internal metric.
A switch or a new method will be added to use the exact algorithm explained in the paper in a few days.



