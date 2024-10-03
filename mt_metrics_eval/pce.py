
# Copyright 2024 Brian Thompson. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np


def compute_pairwise_p_values(seg_scores, num_permutations=1000, seed: int = 4):
    """
    Author: Brian Thompson
    Date: June 2024

    Suppose we have test set consisting of L=5 segments, and two systems, systemsA and systemB,
    for which we have segment-level scores scoresA and scoresB:
       scoresA = [0.8, 0.9, 0.7, 1.0, 0.6]
       scoresB = [0.2, 0.3, 0.1, 0.4, 0.0]

    Typically we would average segment-level scores to get system level scores, but for convenience later on
    we will define system scores to be the sum of segment-level scores. This gives us a delta system-level score of:
        test_delta = sum(scoresA) - sum(scoresB) = 4.0 - 1.0 = 3.0

    To run a paired permutation test, we first generate a new set of scores scores0,
    where each score0[i] is randomly selected from either scoresA[i] or scoresB[i].
    Let's define a random boolean mask:
       m = [1, 0, 0, 1, 1]

    and used it to select scores0:
       scores0 = m.*scoresA + (1-m).*scoresB = [0.8, 0.3, 0.1, 1.0, 0.6]   # selected from [A, B, B, A, A], respectively

    Likewise, we compose scores1 using all the scores which were not selected for scores0:
       scores1 = (1-m).*scoresA + m.*scoresB = [0.2, 0.9, 0.7, 0.4, 0.0]   # selected from [B, A, A, B, B], respectively

    To get the delta system-level score for our two mock systems, we need to compute:
       null_delta = sum(scores0) - sum(scores1)
                  = sum(m.*scoresA + (1-m).*scoresB) - sum((1-m).*scoresA + m.*scoresB)
                  = sum((2m-1).*scoresA) - sum((2m-1).*scoresB
                  = (2m-1) * scoresA.T - (2m-1) * scoresB.T
                  = [ 1, -1, -1,  1,  1] * [[0.8],  -  [ 1, -1, -1,  1,  1] * [[0.2],  =  0.8 - 0.2  =  0.6
                                            [0.9],                             [0.3],
                                            [0.7],                             [0.1],
                                            [1.0],                             [0.4],
                                            [0.6]]                             [0.0]]

    To compute many different permutations, we replace the vector m with a matrix of size (num_permutations, L):
       null_delta = [[ 1,  1, -1, -1, -1], * [[0.8],  -  [[ 1,  1, -1, -1, -1], * [[0.2],  = [[-0.6],  - [[ 0.0],   =  [[-0.6]
                     [ 1, -1,  1, -1,  1],    [0.9],      [ 1, -1,  1, -1,  1],    [0.3],     [ 0.2],     [-0.4],       [ 0.6],
                     [ 1, -1,  1,  1, -1],    [0.7],      [ 1, -1,  1,  1, -1],    [0.1],     [ 1.0],     [ 0.4],       [ 0.6],
                     [-1,  1, -1, -1,  1],    [1.0],      [-1,  1, -1, -1,  1],    [0.4],     [-1.0],     [-0.4],       [-0.6],
                     [ 1,  1,  1, -1,  1],    [0.6]]      [ 1,  1,  1, -1,  1],    [0.0]]     [ 2.0],     [ 0.2],       [ 1.8],
                     [-1,  1, -1,  1, -1],                [-1,  1, -1,  1, -1],               [-0.2],     [ 0.4],       [-0.6],
                     [ 1,  1,  1,  1,  1],                [ 1,  1,  1,  1,  1],               [ 4.0],     [ 1.0],       [ 3.0],
                     [ 1, -1,  1, -1,  1],                [ 1, -1,  1, -1,  1],               [ 0.2],     [-0.4],       [ 0.6],
                     [ 1,  1, -1, -1,  1],                [ 1,  1, -1, -1,  1],               [ 0.6],     [ 0.0],       [ 0.6],
                     [-1,  1, -1, -1, -1]]                [ 1, -1, -1,  1, -1]]               [-2.2]]     [-0.4]]       [-1.8]]

    To test the significance that system A is better than system B, we compute:
       null_delta >= test_delta  =  [[-0.6]  >= 3   =   [[False],
                                     [ 0.6],             [False],
                                     [ 0.6],             [False],
                                     [-0.6],             [False],
                                     [ 1.8],             [False],
                                     [-0.6],             [False],
                                     [ 3.0],             [True ],
                                     [ 0.6],             [False],
                                     [ 0.6],             [False],
                                     [-1.8]]             [False]]

    The p value is the fraction of the time that null_delta >= test_delta, in this case 1/10 = 0.1

    The above discussion was for a single system pair, but we actually need to compute p values for each pairwise
    within a set systems systemA, systemB, ... systemN. In practice, the computation bottleneck is generating
    the random boolean vector m, so we generate m once and use it for all pairs of systems.

    Reusing m also allows us to avoid most of the N^2 computations by pre-computing (2m-1) * scoresA.T,
    (2m-1) * scoresB.T, ..., (2m-1) * scoresN.T.

    Test speed:
    python -m timeit -s "import numpy as np; from pairwise_paired_permutation_test import compute_pairwise_p_values; x=np.random.random(size=(14,1300))" "compute_pairwise_p_values(x, num_permutations=1000)"

    :param seg_scores: segment-level scores, with shape (num_systems, num_segments)
    :param num_permutations: Number of permutations for permutation test
    :param seed: The random seed
    :return: np.array of size (num_systems, num_systems), where the upper triangle has been populated
       with p-values for the hypothesis that system[i] > system[j]
    """
    num_systems, num_segments = seg_scores.shape

    rng = np.random.default_rng(seed)
    # initialize in range [0, 1)
    two_m_minus_one = rng.random(size=(num_permutations, num_segments), dtype=np.float32)
    # quantize to 0 or 1, in place
    np.rint(two_m_minus_one, out=two_m_minus_one, casting='same_kind')
    # scale and shift to get -1.0 and +1.0, in place
    two_m_minus_one *= 2.0
    two_m_minus_one -= 1.0

    seg_scores = seg_scores.astype(np.float32)  # shape: (num_systems, num_segments)
    sys_scores = np.sum(seg_scores, axis=1)  # shape: (num_systems, )

    partial = np.matmul(two_m_minus_one, seg_scores.T)  # shape: (num_permutations, num_systems)

    # initialize p value matrix to NaN
    p_vals = np.empty((num_systems, num_systems,)) * np.nan
    # populate upper triangle
    for ii in range(num_systems):
        for jj in range(ii + 1, num_systems):
            null_delta = partial[:, ii] - partial[:, jj]  # shape: (num_permutations, )
            test_delta = sys_scores[ii] - sys_scores[jj]  # float
            p_vals[ii, jj] = np.sum(null_delta >= test_delta) / num_permutations

    return p_vals


def compute_one_minus_pce(human_pairwise_p_vals, metric_pairwise_p_vals):
    """
    Author: Brian Thompson
    Date: June 2024

    Pairwise Confidence Error (PCE) is the absolute difference between
      the p value for the conclusion that one system is better than another given human judgements and
      the p value for the conclusion for the same system comparison given metric judgements,
      averaged over all system pairings for a set of systems.

    We return 1-PCE to be comparable with pairwise accuracy [i.e. range from 0 to 1, higher is better]

    :param human_pairwise_p_vals: np.array of shape (num_systems, num_systems),
        where the upper triangle has been populated with p-values for system[i] > system[j]
        computed from human judgements
    :param metric_pairwise_p_vals: np.array of shape (num_systems, num_systems),
        where the opper triangle has been populated with p-values for system[i] > system[j]
        computed from metric scores
    :return: 1-PCE
    """
    num_systems = human_pairwise_p_vals.shape[0]
    upper_tri_idxs = np.triu_indices(num_systems, 1)
    return 1.0 - np.mean(np.abs(human_pairwise_p_vals - metric_pairwise_p_vals)[upper_tri_idxs])


