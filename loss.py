from abc import ABC, abstractmethod

from scipy.special import expit, logsumexp
import numpy as np
from numba import njit, prange
import numba


# TODO: Write proper docstrings


@njit
def _get_threads_chunks(total_size):
    # Divide [0, total_size - 1] into n_threads contiguous regions, and
    # returns the starts and ends of each region. Used to simulate a 'static'
    # scheduling.
    n_threads = numba.config.NUMBA_DEFAULT_NUM_THREADS
    sizes = np.full(n_threads, total_size // n_threads, dtype=np.int32)
    sizes[:total_size % n_threads] += 1
    starts = np.zeros(n_threads, dtype=np.int32)
    starts[1:] = np.cumsum(sizes[:-1])
    ends = starts + sizes

    return starts, ends, n_threads



class Loss(ABC):

    def init_gradients_and_hessians(self, n_samples, n_trees_per_iteration):
        shape = (n_samples, n_trees_per_iteration)
        #print(shape)
        gradients = np.zeros(shape=shape, dtype=np.float32)
        if self.hessian_is_constant:
            hessians = np.ones(shape=1, dtype=np.float32)
        else:
            # won't be used anyway #
            hessians = np.empty(shape=shape, dtype=np.float32)

        return gradients, hessians

    @abstractmethod
    def update_gradients_and_hessians(self, gradients, hessians, y_true,
                                      raw_predictions):
        pass


class LeastSquares(Loss):

    hessian_is_constant = True

    def __call__(self, y_true, raw_predictions, average=True):
        # shape (n_samples, 1) --> (n_samples,). reshape(-1) is more likely to
        # return a view.
        raw_predictions = raw_predictions
        loss = np.power(y_true - raw_predictions, 2)
        return loss.mean() if average else loss

    def inverse_link_function(self, raw_predictions):
        return raw_predictions

    def update_gradients_and_hessians(self, gradients, hessians, y_true,
                                      raw_predictions):
        return _update_gradients_least_squares(gradients, y_true,
                                               raw_predictions)


@njit(parallel=True, fastmath=True)
def _update_gradients_least_squares(gradients, y_true, raw_predictions):
    # shape (n_samples, 1) --> (n_samples,). reshape(-1) is more likely to
    # return a view.
    #print('update-grad')
    raw_predictions = raw_predictions
    n_samples = raw_predictions.shape[0]
    starts, ends, n_threads = _get_threads_chunks(total_size=n_samples)
    for thread_idx in prange(n_threads):
        for i in range(starts[thread_idx], ends[thread_idx]):
            # Note: a more correct exp is 2 * (raw_predictions - y_true) but
            # since we use 1 for the constant hessian value (and not 2) this
            # is strictly equivalent for the leaves values.
            gradients[i] = raw_predictions[i] - y_true[i]
    #print('grad', gradients)

_LOSSES = {'least_squares': LeastSquares}
