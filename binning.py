import numpy as np
from numba import njit, prange
from sklearn.utils import check_random_state, check_array
from sklearn.base import BaseEstimator, TransformerMixin


def find_binning_thresholds(data, max_bins=256, subsample=int(2e6),
                            random_state=None):
    """Extract feature-wise equally-spaced quantiles from numerical data

    Subsample the dataset if too large as the feature-wise quantiles
    should be stable.

    If the number of unique values for a given feature is less than
    ``max_bins``, then the unique values are used instead of the quantiles.

    Parameters
    ----------
    data: array-like, shape=(n_samples, n_design, n_features)
        The numerical dataset to analyse.

    max_bins: int, optional (default=256)
        The number of bins to extract for each feature. As we code the binned
        values as 8-bit integers, max_bins should be no larger than 256.

    subsample: int, optional (default=2e6)
        Number of random subsamples to consider to compute the quantiles.

    random_state: int or numpy.random.RandomState or None, \
        optional (default=None)
        Pseudo-random number generator to control the random sub-sampling.

    Return
    ------
    binning_thresholds: tuple of arrays
        For each feature, stores the increasing numeric values that can
        be used to separate the bins.
        len(binning_thresholds) == n_features.
        Each array has size ``(n_bins - 1)`` where:
        ``n_bins == min(max_bins, len(np.unique(data[:, feature_idx])))``
    """
    if not (2 <= max_bins <= 256):
        raise ValueError(f'max_bins={max_bins} should be no smaller than 2 '
                         f'and no larger than 256.')
    rng = check_random_state(random_state)
    n_samples, n_design, n_features = data.shape
    data_ = data.reshape(n_samples*n_design, n_features, order='f')
    if data_.shape[0] > subsample:
        subset = rng.choice(np.arange(data_.shape[0]), subsample)
        data_ = data_[subset]
    dtype = data.dtype
    if dtype.kind != 'f':
        dtype = np.float32

    percentiles = np.linspace(0, 100, num=max_bins + 1)[1:-1]
    binning_thresholds = []
    for f_idx in range(data_.shape[1]):
        col_data = np.ascontiguousarray(data_[:, f_idx], dtype=dtype)
        distinct_values = np.unique(col_data)
        if len(distinct_values) <= max_bins:
            midpoints = (distinct_values[:-1] + distinct_values[1:])
            midpoints *= .5
        else:
            # We sort again the data in this case. We could compute
            # approximate midpoint percentiles using the output of
            # np.unique(col_data, return_counts) instead but this is more
            # work and the performance benefit will be limited because we
            # work on a fixed-size subsample of the full data.
            midpoints = np.percentile(col_data, percentiles,
                                      interpolation='midpoint').astype(dtype)
        binning_thresholds.append(midpoints)
    return tuple(binning_thresholds)


def _map_to_bins(data, binning_thresholds=None, out=None):
    """Bin numerical values to discrete integer-coded levels.

    Parameters
    ----------
    data : array-like, shape=(n_samples, n_design, n_features)
        The numerical data to bin.
    binning_thresholds : tuple of arrays
        For each feature, stores the increasing numeric values that are
        used to separate the bins.
    out : array-like
        If not None, write result inplace in out.

    Returns
    -------
    binned_data : array of int, shape=data.shape
        The binned data.
    """
    # TODO: add support for categorical data encoded as integers
    # TODO: add support for sparse data (numerical or categorical)
    n_samples, n_design, n_features = data.shape
    data_ = data.reshape(n_samples * n_design, n_features, order='f')
    if out is not None:
        assert out.shape == data.shape
        assert out.dtype == np.uint8
        assert out.flags.f_contiguous
        binned = out
    else:
        binned = np.zeros_like(data_, dtype=np.uint8, order='f')

    binning_thresholds = tuple(np.ascontiguousarray(bt, dtype=np.float32)
                               for bt in binning_thresholds)

    for feature_idx in range(data_.shape[1]):
        _map_num_col_to_bins(data_[:, feature_idx],
                             binning_thresholds[feature_idx],
                             binned[:, feature_idx])
    return np.array(binned.reshape(data.shape, order='f'))


@njit(parallel=True)
def _map_num_col_to_bins(data, binning_thresholds, binned):
    """Binary search to the find the bin index for each value in data."""
    for i in prange(data.shape[0]):
        # TODO: add support for missing values (NaN or custom marker)
        left, right = 0, binning_thresholds.shape[0]
        while left < right:
            middle = (right + left - 1) // 2
            if data[i] <= binning_thresholds[middle]:
                right = middle
            else:
                left = middle + 1
        binned[i] = left


class BinMapper(BaseEstimator, TransformerMixin):
    """Transformer that maps a dataset into integer-valued bins

    The bins are created in a feature-wise fashion, with equally-spaced
    quantiles.

    Parameters
    ----------
    max_bins : int, optional (default=256)
        The maximum number of bins to use. If for a given feature the number of
        unique values is less than ``max_bins``, then those unique values
        will be used instead of the quantiles.
    subsample : int, optional (default=1e5)
        If ``n_samples > subsample``, then ``sub_samples`` samples will be
        randomly choosen to compute the quantiles.
        TODO: accept None?
    random_state: int or numpy.random.RandomState or None, \
        optional (default=None)
        Pseudo-random number generator to control the random sub-sampling.
    """
    def __init__(self, max_bins=256, subsample=int(1e6), random_state=None):
        self.max_bins = max_bins
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit data X by computing the binning thresholds.

        Parameters
        ----------
        X: array-like
            The data to bin

        Returns
        -------
        self : object
        """
        n_samples, n_design, n_features = X.shape

        X_ = check_array(X.reshape(n_samples*n_design, n_features, order='f'))
        self.bin_thresholds_ = find_binning_thresholds(
            X, self.max_bins, subsample=self.subsample,
            random_state=self.random_state)

        self.n_bins_per_feature_ = np.array(
            [thresholds.shape[0] + 1 for thresholds in self.bin_thresholds_],
            dtype=np.uint32)

        return self

    def transform(self, X):
        """Bin data X.

        Parameters
        ----------
        X: array-like
            The data to bin

        Returns
        -------
        X_binned : array-like
            The binned data"""
        return _map_to_bins(X, binning_thresholds=self.bin_thresholds_)
