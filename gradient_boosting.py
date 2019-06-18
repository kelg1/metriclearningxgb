"""
Gradient Boosting decision trees for classification and regression.
"""
from abc import ABC, abstractmethod

import numpy as np
from numba import njit, prange, jit
from time import time
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils import check_X_y, check_random_state
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from metriclearningxgb.binning import BinMapper
from metriclearningxgb.grower import TreeGrower
from metriclearningxgb.loss import _LOSSES

class BaseGradientBoostingMachine(BaseEstimator, ABC):
    """Base class for gradient boosting estimators."""

    @abstractmethod
    def __init__(self, loss, learning_rate, max_iter, max_leaf_nodes,
                 max_depth, min_samples_leaf, l2_regularization, max_bins,
                 max_no_improvement, validation_split, scoring, tol, verbose,
                 random_state, ):
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_bins = max_bins
        self.max_no_improvement = max_no_improvement
        self.validation_split = validation_split
        self.scoring = scoring
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

    def _validate_parameters(self):
        """Validate parameters passed to __init__.

        The parameters that are directly passed to the grower are checked in
        TreeGrower."""

        if self.loss not in self._VALID_LOSSES:
            raise ValueError(
                "Loss {} is not supported for {}. Accepted losses"
                "are {}.".format(self.loss, self.__class__.__name__,
                                 ', '.join(self._VALID_LOSSES)))

        if self.learning_rate <= 0:
            raise ValueError(f'learning_rate={self.learning_rate} must '
                             f'be strictly positive')
        if self.max_iter < 1:
            raise ValueError(f'max_iter={self.max_iter} must '
                             f'not be smaller than 1.')
        if self.max_no_improvement < 1:
            raise ValueError(f'max_no_improvement={self.max_no_improvement} '
                             f'must not be smaller than 1.')
        if self.validation_split is not None and self.validation_split <= 0:
            raise ValueError(f'validation_split={self.validation_split} '
                             f'must be strictly positive, or None.')
        if self.tol <= 0:
            raise ValueError(f'tol={self.tol} '
                             f'must be strictly positive.')
    #add#
    @property
    def Xd(self):
        return self._Xd

    @property
    def yd(self):
        return self._yd

    @Xd.setter
    def Xd(self, value):
        self._Xd = value

    @yd.setter
    def yd(self, value):
        self._yd = value

    def fit(self, X, y, Xd, yd):

        """Fit the gradient boosting model.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            The input samples.
        
        Xd : array-like, shape=(n_design, n_features)
            The design samples.

        y : array-like, shape=(n_samples, output_dim=2)
            Target values.

        yd : array-like, shape=(n_samples, output_dim=2)
            Design target values.

        Returns
        -------
        self : object
        """
        #add#
        #if (not hasattr(self, 'Xd')) or (not hasattr(self, 'yd')):
        #    raise NotImplementedError
        
        fit_start_time = time()
        acc_find_split_time = 0.  # time spent finding the best splits
        acc_apply_split_time = 0.  # time spent splitting nodes
        # time spent predicting X for gradient and hessians update
        acc_prediction_time = 0.
        # TODO: add support for mixed-typed (numerical + categorical) data
        # TODO: add support for missing data
        # TODO: add support for pre-binned data (pass-through)?
        # TODO: test input checking

        X, y = check_X_y(X, y, dtype=[np.float32, np.float64],
            ## Add by k## 
            multi_output=True)
        y = self._encode_y(y)
        #add#
        
        rng = check_random_state(self.random_state)
        
        ## Add by k ## 
        Xd_, yd_ = check_X_y(Xd, yd, dtype=[np.float32, np.float64],
            ## Add by k## 
            multi_output=True)
        yd_ = self._encode_y(yd_)
        self.Xd = Xd_ # X design
        self.yd = yd_ # y design 

        self.n_design = self.Xd.shape[0]
        self.output_dim = np.reshape(self.yd, (1, -1)).shape[1] 
        assert(self.Xd.shape[1] == X.shape[1])
        ## Add by k ## 

        self._validate_parameters()

        tic = time()
        self.bin_mapper_ = BinMapper(max_bins=self.max_bins, random_state=rng)
        
        # modify X 
        diff_X_Xd = np.abs(X[:,:, np.newaxis] - self.Xd[:,:, np.newaxis].T).reshape(
            X.shape[0], self.Xd.shape[0], X.shape[1], order='f')
        mean_X_Xd = .5*(X[:,:, np.newaxis] + self.Xd[:,:, np.newaxis].T).reshape(
            X.shape[0], self.Xd.shape[0], X.shape[1], order='f')
        Phi_X_Xd = np.concatenate((diff_X_Xd, mean_X_Xd), axis=2)
        #print(Phi_X_Xd.shape)
        if self.verbose:
            print(f"Binning {Phi_X_Xd.nbytes / 1e9:.3f} GB of data: ", end="",
                  flush=True)
        X_binned = self.bin_mapper_.fit_transform(Phi_X_Xd)
        ## add by k ##
        
        ########
        ## OK ##
        ########

        toc = time()
        if self.verbose:
            duration = toc - tic
            troughput = X.nbytes / duration
            print(f"{duration:.3f} s ({troughput / 1e6:.3f} MB/s)")

        self.loss_ = self._get_loss()

        if self.validation_split is not None:
            # stratify for classification
            stratify = y if hasattr(self.loss_, 'predict_proba') else None
            if hasattr(self.loss_, 'predict_proba'):
                raise(NotImplementedError)
            X_binned_train, X_binned_val, y_train, y_val = train_test_split(
                X_binned, y, test_size=self.validation_split,
                stratify=stratify, random_state=rng)
            X_binned_train = np.asfortranarray(X_binned_train)
            X_binned_val = np.asfortranarray(X_binned_val)
            # Histogram computation is faster on feature-aligned data.
        else:
            X_binned_train, y_train = X_binned, y
            X_binned_val, y_val = None, None
            X_binned_train = np.asfortranarray(X_binned_train)
        
        # Subsample the training set for score-based monitoring.
        subsample_size = 10000
        if X_binned_train.shape[0] < subsample_size:
            X_binned_small_train = np.asfortranarray(X_binned_train)
            y_small_train = y_train
        else:
            indices = rng.choice(
                np.arange(X_binned_train.shape[0]), subsample_size)
            X_binned_small_train = X_binned_train[indices]
            y_small_train = y_train[indices]
        self.X_binned_small_train = X_binned_small_train
        self.X_binned_val = X_binned_val
        if self.verbose:
            print("Fitting gradient boosted rounds:")

        #n_samples = X_binned_train.shape[0] * X_binned_train.shape[1]
        n_samples = X_binned_small_train.shape[0] 
        # values predicted by the trees. Used as-is in regression, and
        # transformed into probas and / or classes for classification
        raw_predictions = np.zeros(shape=(n_samples,
         self.n_trees_per_iteration_),
            dtype=y_train.dtype
        )
        # gradients and hessians are 1D arrays of size
        # n_samples * n_trees_per_iteration

        # gradients and hessians have changed


        gradients, hessians = self.loss_.init_gradients_and_hessians(
            n_samples=n_samples,
            n_trees_per_iteration=self.n_trees_per_iteration_
        )
        #print('raw_', raw_predictions)
        #print('gradient', gradients)
        # predictors_ is a matrix of TreePredictor objects with shape
        # (n_iter_, n_trees_per_iteration)
        self.predictors_ = predictors = []
        self.train_scores_ = []
        if self.validation_split is not None:
            self.validation_scores_ = []
        scorer = check_scoring(self, self.scoring)
        gb_start_time = time()
        # TODO: compute training loss and use it for early stopping if no
        # validation data is provided?
        self.n_iter_ = 0
        while True:
            should_stop = self._stopping_criterion(
                gb_start_time, scorer, X_binned_small_train, y_small_train,
                X_binned_val, y_val)
            if should_stop or self.n_iter_ == self.max_iter:
                break

            # Update gradients and hessians, inplace
            self.loss_.update_gradients_and_hessians(gradients, hessians,
                                                     y_small_train, raw_predictions)
            
            #print('grad', gradients)
            predictors.append([])

            # Build `n_trees_per_iteration` trees.
            for k, (gradients_at_k, hessians_at_k) in enumerate(zip(
                    np.array_split(gradients, self.n_trees_per_iteration_),
                    np.array_split(hessians, self.n_trees_per_iteration_))):
                # the xxxx_at_k arrays are **views** on the original arrays.
                # Note that for binary classif and regressions,
                # n_trees_per_iteration is 1 and xxxx_at_k is equivalent to the
                # whole array.

                #X_binned_small_train_, _, gradients_at_k_, _, indices_subsample_, _ = \
                #train_test_split(X_binned_small_train, gradients_at_k, np.arange(len(X_binned_small_train)), \
                #train_size=subsample_ratio, shuffle=False, random_state=0)
                #X_binned_small_train_ = np.asfortranarray(X_binned_small_train_)
                
                grower = TreeGrower(
                    X_binned_small_train, gradients_at_k, hessians_at_k, yd,
                    max_bins=self.max_bins,
                    n_bins_per_feature=self.bin_mapper_.n_bins_per_feature_,
                    max_leaf_nodes=self.max_leaf_nodes,
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    l2_regularization=self.l2_regularization,
                    shrinkage=self.learning_rate)
                grower.grow()
                #print('I grew')

                acc_apply_split_time += grower.total_apply_split_time
                acc_find_split_time += grower.total_find_split_time

                predictor = grower.make_predictor(
                    bin_thresholds=self.bin_mapper_.bin_thresholds_)
                predictors[-1].append(predictor)

                tic_pred = time()
                # prepare leaves_data so that _update_raw_predictions can be
                # @njitted
                leaves_data = [(l.value, l.sample_indices)
                               for l in grower.finalized_leaves]
                _update_raw_predictions(leaves_data, self.yd, raw_predictions[:, k])
                #print('raw_pred', raw_predictions)
                toc_pred = time()
                acc_prediction_time += toc_pred - tic_pred

            self.n_iter_ += 1
            #self.learning_rate *= 1. # maybe to set 
            self.learning_rate = 1./(self.n_iter_+1)
            #self.max_depth += 1
            #self.max_depth = min(5, self.max_depth)
            #print('pred', raw_predictions)
            #print('n_iter', self.n_iter_)
        if self.verbose:
            duration = time() - fit_start_time
            n_total_leaves = sum(
                predictor.get_n_leaf_nodes()
                for predictors_at_ith_iteration in self.predictors_
                for predictor in predictors_at_ith_iteration)
            n_predictors = sum(
                len(predictors_at_ith_iteration)
                for predictors_at_ith_iteration in self.predictors_)
            print(f"Fit {n_predictors} trees in {duration:.3f} s, "
                  f"({n_total_leaves} total leaves)")
            print(f"{'Time spent finding best splits:':<32} "
                  f"{acc_find_split_time:.3f}s")
            print(f"{'Time spent applying splits:':<32} "
                  f"{acc_apply_split_time:.3f}s")
            print(f"{'Time spent predicting:':<32} "
                  f"{acc_prediction_time:.3f}s")
        self.train_scores_ = np.asarray(self.train_scores_)
        if self.validation_split is not None:
            self.validation_scores_ = np.asarray(self.validation_scores_)
        return self

    def _raw_predict(self, X):
        """Return the sum of the leaves values over all predictors.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            The input samples.

        Returns
        -------
        raw_predictions : array, shape (n_samples * n_trees_per_iteration,)
            The raw predicted values.
        """
        # TODO: check input / check_fitted
        # TODO: make predictor behave correctly on pre-binned data
        n_samples = X.shape[0]
        n_design = X.shape[1]
        n_features = X.shape[2]
        is_binned = X.dtype == np.uint8
        #print('is_binned', is_binned)
        raw_predictions = np.zeros(shape=(n_samples, self.n_trees_per_iteration_),
                             dtype=np.float32)
        # Should we parallelize this?
        for predictors_of_ith_iteration in self.predictors_:
            for k, predictor in enumerate(predictors_of_ith_iteration):
                start, end = n_samples * k, n_samples * (k + 1)
                #print('pred', predictor.predict_binned(X_binned).shape)
                pred = predictor.predict_binned if is_binned else predictor.predict
                #print('pred', pred, '\n')
                #print(self.yd.shape)
                pred_X = np.dot(pred(X), self.yd)

                #print('predict', pred)
                #print(predicted.shape)
                #print(predicted[start:end])
                raw_predictions[start:end,k] += pred_X.reshape(raw_predictions[start:end, k].shape, order='f')
        return raw_predictions

    def _predict_binned(self, X_binned):
        """Predict values or classes for binned data X.

        TODO: This is incorrect now that we support classification. This
        should return classes, not the raw values from the leaves.
        Check back shapes and names once we fix this.

        Parameters
        ----------
        X_binned : array-like, shape=(n_samples, n_features)
            The binned input samples. Entries should be integers.

        Returns
        -------
        y : array, shape (n_samples * n_trees_per_iteration)
            The predicted values or classes.
        """
        #print('predict_binned_')
        n_samples = X_binned.shape[0]
        n_design = X_binned.shape[1]
        #dim_output = np.reshape(self.yd, (-1,1)).shape[1]
        n_features = X_binned.shape[2]
        
        predicted = np.zeros(shape=(n_samples, self.n_trees_per_iteration_),
                             dtype=np.float32)
        # Should we parallelize this?
        for predictors_of_ith_iteration in self.predictors_:
            for k, predictor in enumerate(predictors_of_ith_iteration):
                start, end = n_samples * k, n_samples * (k + 1)
                #print('pred', predictor.predict_binned(X_binned).shape)
                pred = predictor.predict_binned(X_binned)
                #print('pred', pred.shape)
                #print(self.yd.shape)
                pred_ = np.dot(pred, self.yd)
                #print('pred_yd', pred_.shape)
                #print('predicted', predicted.shape)
                #print('k', k)
                #print('pred_', pred_.shape)
                #print('predcc', predicted[start:end, k].shape)
                predicted[start:end, k] += pred_.reshape(predicted[start:end, k].shape, order='f')
        return predicted

    def _stopping_criterion(self, start_time, scorer, X_binned_train, y_train,
                            X_binned_val, y_val):
        log_msg = f"[{self.n_iter_}/{self.max_iter}]"

        if self.scoring is not None:
            # TODO: make sure that self.predict can work on binned data and
            # then only use the public scorer.__call__.
            predicted_train = self._predict_binned(X_binned_train)
            
            score_train = scorer._score_func(y_train, predicted_train)
            self.train_scores_.append(score_train)
            log_msg += f" {self.scoring} train: {score_train:.5f},"

            if self.validation_split is not None:
                predicted_val = self._predict_binned(X_binned_val)
                score_val = scorer._score_func(y_val, predicted_val)
                self.validation_scores_.append(score_val)
                log_msg += f", {self.scoring} val: {score_val:.5f},"

        if self.n_iter_ > 0:
            # TODO: that's rather the average iteration time right?
            # Also, all of the logging/printing should be done somewhere else
            iteration_time = (time() - start_time) / self.n_iter_
            predictors_of_ith_iteration = [
                predictors_list for predictors_list in self.predictors_[-1]
                if predictors_list
            ]
            n_trees = len(predictors_of_ith_iteration)
            max_depth = max(predictor.get_max_depth()
                            for predictor in predictors_of_ith_iteration)
            n_leaves = sum(predictor.get_n_leaf_nodes()
                           for predictor in predictors_of_ith_iteration)

            if n_trees == 1:
                log_msg += (f" {n_trees} tree, {n_leaves} leaves, ")
            else:
                log_msg += (f" {n_trees} trees, {n_leaves} leaves ")
                log_msg += (f"({int(n_leaves / n_trees)} on avg), ")

            log_msg += (f"max depth = {max_depth} "
                        f"in {iteration_time:0.3f}s")

        if self.verbose:
            print(log_msg)

        if self.validation_split is not None:
            return self._should_stop(self.validation_scores_)
        else:
            return self._should_stop(self.train_scores_)

    def _should_stop(self, scores):
        if (len(scores) == 0 or
                (self.max_no_improvement
                 and len(scores) < self.max_no_improvement)):
            return False
        context_scores = scores[-self.max_no_improvement:]
        candidate = scores[-self.max_no_improvement]
        tol = 0. if self.tol is None else self.tol
        # sklearn scores: higher is always better.
        best_with_tol = min(context_scores) * (1 + tol)
        return candidate <= best_with_tol

    @abstractmethod
    def _get_loss(self):
        pass

    @abstractmethod
    def _encode_y(self, y=None):
        pass




class GradientBoostingRegressor(BaseGradientBoostingMachine, RegressorMixin):
    """Scikit-learn compatible Gradient Boosting Tree for regression.

    Parameters
    ----------
    loss : {'least_squares'}, optional(default='least_squares')
        The loss function to use in the boosting process.
    learning_rate : float, optional(default=TODO)
        The learning rate, also known as *shrinkage*. This is used as a
        multiplicative factor for the leaves values.
    max_iter : int, optional(default=TODO)
        The maximum number of iterations of the boosting process, i.e. the
        maximum number of trees.
    max_leaf_nodes : int, optional(default=TODO)
        The maximum number of leaves for each tree.
    max_depth : int, optional(default=TODO)
        The maximum depth of each tree. The depth of a tree is the number of
        nodes to go from the root to the deepest leaf.
    min_samples_leaf : int, optional(default=TODO)
        The minimum number of samples per leaf.
    l2_regularization : float, optional(default=TODO)
        The L2 regularization parameter.
    max_bins : int, optional(default=256)
        The maximum number of bins to use. Before training, each feature of
        the input array ``X`` is binned into at most ``max_bins`` bins, which
        allows for a much faster training stage. Features with a small
        number of unique values may use less than ``max_bins`` bins. Must be no
        larger than 256.
    max_no_improvement : int, optional(default=TODO)
        TODO
    validation_split : int or float, optional(default=TODO)
        TODO
    scoring : str, optional(default=TODO)
        TODO
    verbose: int, optional(default=0)
        The verbosity level. If not zero, print some information about the
        fitting process.
    random_state : int, np.random.RandomStateInstance or None, \
        optional(default=None)
        TODO: any chance we can link to sklearn glossary?
    """

    _VALID_LOSSES = ('least_squares',)

    def __init__(self, loss='least_squares', learning_rate=1, max_iter=50,
                 max_leaf_nodes=31, max_depth=3, min_samples_leaf=5,
                 l2_regularization=1, max_bins=3, max_no_improvement=3,
                 validation_split=None, scoring='neg_mean_squared_error',
                 tol=1e-7, verbose=1, random_state=42):
        super(GradientBoostingRegressor, self).__init__(
            loss=loss, learning_rate=learning_rate, max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization, max_bins=max_bins,
            max_no_improvement=max_no_improvement,
            validation_split=validation_split, scoring=scoring, tol=tol,
            verbose=verbose, random_state=random_state)

    def predict(self, X):
        """Predict values for X.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array, shape (n_samples,)
            The predicted values.
        """
        # Return raw predictions after converting shape
        # (n_samples, 1) to (n_samples,)
        return self._raw_predict(X).ravel()

    def _encode_y(self, y):
        # Just convert y to float32
        self.n_trees_per_iteration_ = 1
        y = y.astype(np.float32, copy=False)
        return y

    def _get_loss(self):
        return _LOSSES[self.loss]()


#@jit(parallel=True)
def _update_raw_predictions(leaves_data, yd, raw_predictions):
    """Update raw_predictions by reading the predictions of the ith tree
    directly from the leaves.

    Can only be used for predicting the training data. raw_predictions
    contains the sum of the tree values from iteration 0 to i - 1. This adds
    the predictions of the ith tree to raw_predictions.

    Parameters
    ----------
    leaves_data: list of tuples (leaf.value, leaf.sample_indices)
        The leaves data used to update raw_predictions.
    raw_predictions : array-like, shape=(n_samples,)
        The raw predictions for the training data.
    """
    n_design = len(yd)
    for leaf_idx, (leaf_value, sample_indices) in enumerate(leaves_data):
        for i, j in sample_indices:
            #print('leaf_va', leaf_value)
            raw_predictions[i] += leaf_value*yd[j]










