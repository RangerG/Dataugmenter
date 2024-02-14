import numpy as np
from tqdm import tqdm
from ..data_augmenter.Transformation_Based_Methods import jitter, window_slice


def spawner(x, labels, sigma=0.05, verbose=0):
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6983028/
    # use verbose=-1 to turn off warnings
    # use verbose=1 to print out figures

    from ..utils import dtw as dtw
    random_points = np.random.randint(low=1, high=x.shape[1] - 1, size=x.shape[0])
    window = np.ceil(x.shape[1] / 10.).astype(int)
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels

    ret = np.zeros_like(x)
    for i, pat in enumerate(tqdm(x)):
        # guarentees that same one isn‘t selected
        choices = np.delete(np.arange(x.shape[0]), i)
        # remove ones of different classes
        choices = np.where(l[choices] == l[i])[0]
        if choices.size > 0:
            random_sample = x[np.random.choice(choices)]
            # SPAWNER splits the path into two randomly
            path1 = dtw.dtw(pat[:random_points[i]], random_sample[:random_points[i]], dtw.RETURN_PATH,
                            slope_constraint="symmetric", window=window)
            path2 = dtw.dtw(pat[random_points[i]:], random_sample[random_points[i]:], dtw.RETURN_PATH,
                            slope_constraint="symmetric", window=window)
            combined = np.concatenate((np.vstack(path1), np.vstack(path2 + random_points[i])), axis=1)
            if verbose:
                print(random_points[i])
                dtw_value, cost, DTW_map, path = dtw.dtw(pat, random_sample, return_flag=dtw.RETURN_ALL,
                                                         slope_constraint=slope_constraint, window=window)
                dtw.draw_graph1d(cost, DTW_map, path, pat, random_sample)
                dtw.draw_graph1d(cost, DTW_map, combined, pat, random_sample)
            mean = np.mean([pat[combined[0]], random_sample[combined[1]]], axis=0)
            for dim in range(x.shape[2]):
                ret[i, :, dim] = np.interp(orig_steps, np.linspace(0, x.shape[1] - 1., num=mean.shape[0]),
                                           mean[:, dim]).T
        else:
            if verbose > -1:
                print("There is only one pattern of class %d, skipping pattern average" % l[i])
            ret[i, :] = pat
    return jitter(ret, sigma=sigma)


def wdba(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True, verbose=0):
    # https://ieeexplore.ieee.org/document/8215569
    # use verbose = -1 to turn off warnings
    # slope_constraint is for DTW. "symmetric" or "asymmetric"

    from ..utils import dtw as dtw

    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels

    ret = np.zeros_like(x)
    for i in tqdm(range(ret.shape[0])):
        # get the same class as i
        choices = np.where(l == l[i])[0]
        if choices.size > 0:
            # pick random intra-class pattern
            k = min(choices.size, batch_size)
            random_prototypes = x[np.random.choice(choices, k, replace=False)]

            # calculate dtw between all
            dtw_matrix = np.zeros((k, k))
            for p, prototype in enumerate(random_prototypes):
                for s, sample in enumerate(random_prototypes):
                    if p == s:
                        dtw_matrix[p, s] = 0.
                    else:
                        dtw_matrix[p, s] = dtw.dtw(prototype, sample, dtw.RETURN_VALUE,
                                                   slope_constraint=slope_constraint, window=window)

            # get medoid
            medoid_id = np.argsort(np.sum(dtw_matrix, axis=1))[0]
            nearest_order = np.argsort(dtw_matrix[medoid_id])
            medoid_pattern = random_prototypes[medoid_id]

            # start weighted DBA
            average_pattern = np.zeros_like(medoid_pattern)
            weighted_sums = np.zeros((medoid_pattern.shape[0]))
            for nid in nearest_order:
                if nid == medoid_id or dtw_matrix[medoid_id, nearest_order[1]] == 0.:
                    average_pattern += medoid_pattern
                    weighted_sums += np.ones_like(weighted_sums)
                else:
                    path = dtw.dtw(medoid_pattern, random_prototypes[nid], dtw.RETURN_PATH,
                                   slope_constraint=slope_constraint, window=window)
                    dtw_value = dtw_matrix[medoid_id, nid]
                    warped = random_prototypes[nid, path[1]]
                    weight = np.exp(np.log(0.5) * dtw_value / dtw_matrix[medoid_id, nearest_order[1]])
                    average_pattern[path[0]] += weight * warped
                    weighted_sums[path[0]] += weight

            ret[i, :] = average_pattern / weighted_sums[:, np.newaxis]
        else:
            if verbose > -1:
                print("There is only one pattern of class %d, skipping pattern average" % l[i])
            ret[i, :] = x[i]
    return ret


def random_guided_warp(x, labels, slope_constraint="symmetric", use_window=True, dtw_type="normal", verbose=0):
    # use verbose = -1 to turn off warnings
    # slope_constraint is for DTW. "symmetric" or "asymmetric"
    # dtw_type is for shapeDTW or DTW. "normal" or "shape"

    from ..utils import dtw as dtw

    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels

    ret = np.zeros_like(x)
    for i, pat in enumerate(tqdm(x)):
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)
        # remove ones of different classes
        choices = np.where(l[choices] == l[i])[0]
        if choices.size > 0:
            # pick random intra-class pattern
            random_prototype = x[np.random.choice(choices)]

            if dtw_type == "shape":
                path = dtw.shape_dtw(random_prototype, pat, dtw.RETURN_PATH, slope_constraint=slope_constraint,
                                     window=window)
            else:
                path = dtw.dtw(random_prototype, pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)

            # Time warp
            warped = pat[path[1]]
            for dim in range(x.shape[2]):
                ret[i, :, dim] = np.interp(orig_steps, np.linspace(0, x.shape[1] - 1., num=warped.shape[0]),
                                           warped[:, dim]).T
        else:
            if verbose > -1:
                print("There is only one pattern of class %d, skipping timewarping" % l[i])
            ret[i, :] = pat
    return ret


def random_guided_warp_shape(x, labels, slope_constraint="symmetric", use_window=True):
    return random_guided_warp(x, labels, slope_constraint, use_window, dtw_type="shape")


def discriminative_guided_warp(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True,
                               dtw_type="normal", use_variable_slice=True, verbose=0):
    # use verbose = -1 to turn off warnings
    # slope_constraint is for DTW. "symmetric" or "asymmetric"
    # dtw_type is for shapeDTW or DTW. "normal" or "shape"

    from ..utils import dtw as dtw

    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels

    positive_batch = np.ceil(batch_size / 2).astype(int)
    negative_batch = np.floor(batch_size / 2).astype(int)

    ret = np.zeros_like(x)
    warp_amount = np.zeros(x.shape[0])
    for i, pat in enumerate(tqdm(x)):
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)

        # remove ones of different classes
        positive = np.where(l[choices] == l[i])[0]
        negative = np.where(l[choices] != l[i])[0]

        if positive.size > 0 and negative.size > 0:
            pos_k = min(positive.size, positive_batch)
            neg_k = min(negative.size, negative_batch)
            positive_prototypes = x[np.random.choice(positive, pos_k, replace=False)]
            negative_prototypes = x[np.random.choice(negative, neg_k, replace=False)]

            # vector embedding and nearest prototype in one
            pos_aves = np.zeros((pos_k))
            neg_aves = np.zeros((pos_k))
            if dtw_type == "shape":
                for p, pos_prot in enumerate(positive_prototypes):
                    for ps, pos_samp in enumerate(positive_prototypes):
                        if p != ps:
                            pos_aves[p] += (1. / (pos_k - 1.)) * dtw.shape_dtw(pos_prot, pos_samp, dtw.RETURN_VALUE,
                                                                               slope_constraint=slope_constraint,
                                                                               window=window)
                    for ns, neg_samp in enumerate(negative_prototypes):
                        neg_aves[p] += (1. / neg_k) * dtw.shape_dtw(pos_prot, neg_samp, dtw.RETURN_VALUE,
                                                                    slope_constraint=slope_constraint, window=window)
                selected_id = np.argmax(neg_aves - pos_aves)
                path = dtw.shape_dtw(positive_prototypes[selected_id], pat, dtw.RETURN_PATH,
                                     slope_constraint=slope_constraint, window=window)
            else:
                for p, pos_prot in enumerate(positive_prototypes):
                    for ps, pos_samp in enumerate(positive_prototypes):
                        if p != ps:
                            pos_aves[p] += (1. / (pos_k - 1.)) * dtw.dtw(pos_prot, pos_samp, dtw.RETURN_VALUE,
                                                                         slope_constraint=slope_constraint,
                                                                         window=window)
                    for ns, neg_samp in enumerate(negative_prototypes):
                        neg_aves[p] += (1. / neg_k) * dtw.dtw(pos_prot, neg_samp, dtw.RETURN_VALUE,
                                                              slope_constraint=slope_constraint, window=window)
                selected_id = np.argmax(neg_aves - pos_aves)
                path = dtw.dtw(positive_prototypes[selected_id], pat, dtw.RETURN_PATH,
                               slope_constraint=slope_constraint, window=window)

            # Time warp
            warped = pat[path[1]]
            warp_path_interp = np.interp(orig_steps, np.linspace(0, x.shape[1] - 1., num=warped.shape[0]), path[1])
            warp_amount[i] = np.sum(np.abs(orig_steps - warp_path_interp))
            for dim in range(x.shape[2]):
                ret[i, :, dim] = np.interp(orig_steps, np.linspace(0, x.shape[1] - 1., num=warped.shape[0]),
                                           warped[:, dim]).T
        else:
            if verbose > -1:
                print("There is only one pattern of class %d" % l[i])
            ret[i, :] = pat
            warp_amount[i] = 0.
    if use_variable_slice:
        max_warp = np.max(warp_amount)
        if max_warp == 0:
            # unchanged
            ret = window_slice(ret, reduce_ratio=0.9)
        else:
            for i, pat in enumerate(ret):
                # Variable Sllicing
                ret[i] = window_slice(pat[np.newaxis, :, :], reduce_ratio=0.9 + 0.1 * warp_amount[i] / max_warp)[0]
    return ret


def discriminative_guided_warp_shape(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True):
    return discriminative_guided_warp(x, labels, batch_size, slope_constraint, use_window, dtw_type="shape")
