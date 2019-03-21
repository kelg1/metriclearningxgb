# metriclearningxgb

Experimental Gradient Boosting Machines in Python.

The goal of this project is to evaluate whether it's possible to implement a
pure Python yet efficient version histogram-binning of Gradient Boosting
Trees (possibly with all the LightGBM optimizations) while staying in pure
Python 3.6+ using the [numba](http://numba.pydata.org/) jit compiler.

pygbm provides a set of scikit-learn compatible estimator classes that
should play well with the scikit-learn `Pipeline` and model selection tools
(grid search and randomized hyperparameter search).

Longer term plans include integration with dask and dask-ml for
out-of-core and distributed fitting on a cluster.

## Installation

The project is available on PyPI and can be installed with `pip`:

    pip install metriclearningxgb

You'll need Python 3.6 at least.

## Benchmarking


## Acknowledgements


The work from Nicolas Hug is supported by the National Science Foundation
under Grant No. 1740305 and by DARPA under Grant No. DARPA-BAA-16-51

The work from Olivier Grisel is supported by the [scikit-learn initiative
and its partners at Inria Fondation](https://scikit-learn.fondation-inria.fr/en/)