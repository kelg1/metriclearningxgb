import numpy as np
from numba import njit

HISTOGRAM_DTYPE = np.dtype([
    ('sum_gradients', np.float32),
    ('sum_hessians', np.float32),
    ('count', np.uint32),
    ('sample_indices', type(set()))
])


#@njit
def _build_histogram_naive(n_bins, sample_indices, binned_feature,
                           gradients, yd):
    #print('new_histo_building')
    histogram = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    for i, sample_idx in enumerate(sample_indices):
       
       bin_idx = binned_feature[tuple(sample_idx)]
       #print('bin_idx', bin_idx)
       #print('sample_idx', sample_idx)
       #print('g_shape', gradients.shape)
       #print('yd_shape', yd.shape)
       histogram[bin_idx]['sum_gradients'] += np.dot(gradients.ravel()[sample_idx[0]], yd[sample_idx[1]])
       #histogram[bin_idx]['sum_hessians'] += ordered_hessians[i]
       #histogram[bin_idx]['sum_hessians'].append(tuple(sample_idx))
       ll = set(histogram[bin_idx]['sample_indices']) if histogram[bin_idx]['sample_indices'] != 0 else set({})
       ll.add(tuple(sample_idx))
       histogram[bin_idx]['sample_indices'] = ll
       histogram[bin_idx]['count'] += 1

    # n_node_samples = sample_indices.shape[0]
    # unrolled_upper = (n_node_samples // 4) * 4

    # for i in range(0, unrolled_upper, 4):

    #     sample_idx_0 = tuple(sample_indices[i])
    #     sample_idx_1 = tuple(sample_indices[i+1])
    #     sample_idx_2 = tuple(sample_indices[i+2])
    #     sample_idx_3 = tuple(sample_indices[i+3])
        
    #     bin_0 = binned_feature[sample_idx_0]
    #     bin_1 = binned_feature[sample_idx_1]
    #     bin_2 = binned_feature[sample_idx_2]
    #     bin_3 = binned_feature[sample_idx_3]

    #     histogram[bin_0]['sum_gradients'] += np.dot(gradients.ravel()[sample_idx_0[0]], yd[sample_idx_0[1]])
    #     histogram[bin_1]['sum_gradients'] += np.dot(gradients.ravel()[sample_idx_1[0]], yd[sample_idx_1[1]])
    #     histogram[bin_2]['sum_gradients'] += np.dot(gradients.ravel()[sample_idx_2[0]], yd[sample_idx_2[1]])
    #     histogram[bin_3]['sum_gradients'] += np.dot(gradients.ravel()[sample_idx_3[0]], yd[sample_idx_3[1]])
        
    #     ll_0 = set(histogram[bin_0]['sample_indices']) if set(histogram[bin_0]['sample_indices']) != 0 else set({})
    #     ll_0.add(tuple(sample_idx_0))
    #     histogram[bin_0]['sample_indices'] = np.array(list(ll_0))
    #     ll_1 = set(histogram[bin_1]['sample_indices']) if set(histogram[bin_1]['sample_indices']) != 0 else set({})
    #     ll_1.add(tuple(sample_idx_1))
    #     histogram[bin_1]['sample_indices'] = np.array(list(ll_1))
    #     ll_2 = set(histogram[bin_2]['sample_indices']) if set(histogram[bin_2]['sample_indices']) != 0 else set({})
    #     ll_2.add(tuple(sample_idx_2))
    #     histogram[bin_2]['sample_indices'] = np.array(list(ll_2))
    #     ll_3 = set(histogram[bin_3]['sample_indices']) if set(histogram[bin_3]['sample_indices']) != 0 else set({})
    #     ll_3.add(tuple(sample_idx_3))
    #     histogram[bin_3]['sample_indices'] = np.array(list(ll_3))


    #     histogram[bin_0]['count'] += 1
    #     histogram[bin_1]['count'] += 1
    #     histogram[bin_2]['count'] += 1
    #     histogram[bin_3]['count'] += 1

    # for i in range(unrolled_upper, n_node_samples):
    #     sample_idx = sample_indices[i]
    #     bin_idx = binned_feature[tuple(sample_idx)]
    #     #print('bin_idx', bin_idx)
    #     #print('sample_idx', sample_idx)
    #     #print('g_shape', gradients.shape)
    #     #print('yd_shape', yd.shape)
    #     histogram[bin_idx]['sum_gradients'] += np.dot(gradients.ravel()[sample_idx[0]], yd[sample_idx[1]])
    #     #histogram[bin_idx]['sum_hessians'] += ordered_hessians[i]
    #     #histogram[bin_idx]['sum_hessians'].append(tuple(sample_idx))
    #     ll = set(histogram[bin_idx]['sample_indices']) if set(histogram[bin_idx]['sample_indices']) != 0 else set({})
    #     ll.add(tuple(sample_idx))
    #     histogram[bin_idx]['sample_indices'] = np.array(list(ll))
    #     histogram[bin_idx]['count'] += 1

    #print('histo', histogram)
    return histogram

#@njit
def _subtract_histograms(n_bins, hist_a, hist_b):
    """Return hist_a - hist_b"""

    histogram = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    #print('subtraction_histo')
    sg = 'sum_gradients'
    sh = 'sample_indices'
    c = 'count'
    #print('look_here_', hist_a, hist_b)
    for i in range(n_bins):
        histogram[i][sg] = hist_a[i][sg] - hist_b[i][sg]

        try:
            ll = set(hist_a[i][sh]) - set(hist_b[i][sh])
            histogram[i][sh] = ll
        except TypeError:
            histogram[i][sh] = hist_a[i][sh]
        
        histogram[i][c] = hist_a[i][c] - hist_b[i][c]
    #print('look_prob_here', histogram )
    return histogram

#@njit
def _build_histogram_root_no_hessian_naive(n_bins, binned_feature, all_gradients, yd):
    """Special case for the root node

    The root node has to find the a split among all the samples from the
    training set. binned_feature and all_gradients already have a consistent
    ordering.
    """
    histogram = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    n_samples = binned_feature.shape[0]
    n_design = binned_feature.shape[1]
    I,J = np.meshgrid(np.arange(n_samples), np.arange(n_design))
    for i, j in zip(I.ravel(), J.ravel()):
        bin_idx = binned_feature[(i,j)]
        #print(bin_idx)
        histogram[bin_idx]['sum_gradients'] += np.dot(all_gradients.ravel()[i], yd[j])
        #histogram[bin_idx]['sum_hessians'] += ordered_hessians[i]
        #histogram[bin_idx]['sum_hessians'].append(tuple(sample_idx))
        ll = set(histogram[bin_idx]['sample_indices']) if histogram[bin_idx]['sample_indices'] != 0 else set({})
        ll.add(tuple((i,j)))
        histogram[bin_idx]['sample_indices'] = ll
        histogram[bin_idx]['count'] += 1
    #print('hist', histogram)
    return histogram

#@njit
def _build_histogram_root_no_hessian(n_bins, binned_feature, all_gradients):
    """Special case for the root node

    The root node has to find the a split among all the samples from the
    training set. binned_feature and all_gradients already have a consistent
    ordering.
    """
    histogram = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    n_node_samples = binned_feature.shape[0]
    unrolled_upper = (n_node_samples // 4) * 4

    for i in range(0, unrolled_upper, 4):
        bin_0 = binned_feature[i]
        bin_1 = binned_feature[i + 1]
        bin_2 = binned_feature[i + 2]
        bin_3 = binned_feature[i + 3]

        histogram[bin_0]['sum_gradients'] += all_gradients[i]
        histogram[bin_1]['sum_gradients'] += all_gradients[i + 1]
        histogram[bin_2]['sum_gradients'] += all_gradients[i + 2]
        histogram[bin_3]['sum_gradients'] += all_gradients[i + 3]

        histogram[bin_0]['count'] += 1
        histogram[bin_1]['count'] += 1
        histogram[bin_2]['count'] += 1
        histogram[bin_3]['count'] += 1

    for i in range(unrolled_upper, n_node_samples):
        bin_idx = binned_feature[i]
        histogram[bin_idx]['sum_gradients'] += all_gradients[i]
        histogram[bin_idx]['count'] += 1

    return histogram

#@njit
def _build_histogram(n_bins, sample_indices, binned_feature, ordered_gradients,
                     ordered_hessians):
    # Not implemented yet. Use _build_histogram_naive instead. 
    histogram = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    n_node_samples = sample_indices.shape[0]
    unrolled_upper = (n_node_samples // 4) * 4

    for i in range(0, unrolled_upper, 4):
        bin_0 = binned_feature[sample_indices[i]]
        bin_1 = binned_feature[sample_indices[i + 1]]
        bin_2 = binned_feature[sample_indices[i + 2]]
        bin_3 = binned_feature[sample_indices[i + 3]]

        histogram[bin_0]['sum_gradients'] += ordered_gradients[i]
        histogram[bin_1]['sum_gradients'] += ordered_gradients[i + 1]
        histogram[bin_2]['sum_gradients'] += ordered_gradients[i + 2]
        histogram[bin_3]['sum_gradients'] += ordered_gradients[i + 3]

        histogram[bin_0]['sum_hessians'] += ordered_hessians[i]
        histogram[bin_1]['sum_hessians'] += ordered_hessians[i + 1]
        histogram[bin_2]['sum_hessians'] += ordered_hessians[i + 2]
        histogram[bin_3]['sum_hessians'] += ordered_hessians[i + 3]

        histogram[bin_0]['count'] += 1
        histogram[bin_1]['count'] += 1
        histogram[bin_2]['count'] += 1
        histogram[bin_3]['count'] += 1

    for i in range(unrolled_upper, n_node_samples):
        bin_idx = binned_feature[sample_indices[i]]
        histogram[bin_idx]['sum_gradients'] += ordered_gradients[i]
        histogram[bin_idx]['sum_hessians'] += ordered_hessians[i]
        histogram[bin_idx]['count'] += 1

    return histogram


#@njit
def _build_histogram_no_hessian(n_bins, sample_indices, binned_feature,
                                ordered_gradients):
    histogram = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    n_node_samples = sample_indices.shape[0]
    unrolled_upper = (n_node_samples // 4) * 4

    for i in range(0, unrolled_upper, 4):
        bin_0 = binned_feature[sample_indices[i]]
        bin_1 = binned_feature[sample_indices[i + 1]]
        bin_2 = binned_feature[sample_indices[i + 2]]
        bin_3 = binned_feature[sample_indices[i + 3]]

        histogram[bin_0]['sum_gradients'] += ordered_gradients[i]
        histogram[bin_1]['sum_gradients'] += ordered_gradients[i + 1]
        histogram[bin_2]['sum_gradients'] += ordered_gradients[i + 2]
        histogram[bin_3]['sum_gradients'] += ordered_gradients[i + 3]

        histogram[bin_0]['count'] += 1
        histogram[bin_1]['count'] += 1
        histogram[bin_2]['count'] += 1
        histogram[bin_3]['count'] += 1

    for i in range(unrolled_upper, n_node_samples):
        bin_idx = binned_feature[sample_indices[i]]
        histogram[bin_idx]['sum_gradients'] += ordered_gradients[i]
        histogram[bin_idx]['count'] += 1

    return histogram


#@njit
def _build_histogram_root(n_bins, binned_feature, all_gradients,
                          all_hessians):
    """Special case for the root node

    The root node has to find the a split among all the samples from the
    training set. binned_feature and all_gradients already have a consistent
    ordering.
    """
    histogram = np.zeros(n_bins, dtype=HISTOGRAM_DTYPE)
    n_node_samples = binned_feature.shape[0]
    unrolled_upper = (n_node_samples // 4) * 4

    for i in range(0, unrolled_upper, 4):
        bin_0 = binned_feature[i]
        bin_1 = binned_feature[i + 1]
        bin_2 = binned_feature[i + 2]
        bin_3 = binned_feature[i + 3]

        histogram[bin_0]['sum_gradients'] += all_gradients[i]
        histogram[bin_1]['sum_gradients'] += all_gradients[i + 1]
        histogram[bin_2]['sum_gradients'] += all_gradients[i + 2]
        histogram[bin_3]['sum_gradients'] += all_gradients[i + 3]

        histogram[bin_0]['sum_hessians'] += all_hessians[i]
        histogram[bin_1]['sum_hessians'] += all_hessians[i + 1]
        histogram[bin_2]['sum_hessians'] += all_hessians[i + 2]
        histogram[bin_3]['sum_hessians'] += all_hessians[i + 3]

        histogram[bin_0]['count'] += 1
        histogram[bin_1]['count'] += 1
        histogram[bin_2]['count'] += 1
        histogram[bin_3]['count'] += 1

    for i in range(unrolled_upper, n_node_samples):
        bin_idx = binned_feature[i]
        histogram[bin_idx]['sum_gradients'] += all_gradients[i]
        histogram[bin_idx]['sum_hessians'] += all_hessians[i]
        histogram[bin_idx]['count'] += 1

    return histogram
