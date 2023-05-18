# coding=utf-8
# Copyright 2021 Google LLC
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
"""Correlations and related functions."""

import itertools
import math
from typing import Callable, Optional, Tuple
import warnings
import numpy as np
from numpy.typing import ArrayLike
import scipy.stats


class Correlation:
  """Data and algorithms for computing correlations.

  Assuming you've created a Correlation object 'corr', here are the ways you can
  call the various correlation functions (using Pearson as an example, but they
  all work the same way except KendallLike). Adopting the convention from
  the 'average_by' argument to data.CompareMetrics: none = flatten score
  matrices into vectors, item = average over correlations for vectors of scores
  for the same item across all systems; sys =  average over the correlations for
  vectors of scores for the same system across all items:

     level   average_by   call
     sys     ---          corr.Pearson()
     seg     none         corr.Pearson()
     seg     item         corr.Pearson(True, False)
     seg     sys          corr.Pearson(True, True)

  See dosctrings for CorrFunction and KendallLike for further notes on averaging
  and handling gold scores that contain missing entries.
  """

  def __init__(self, num_sys, gold_scores, metric_scores):
    """Construct from parallel vectors that group scores by system."""
    self.num_sys = num_sys
    self.num_items = len(gold_scores) // num_sys  # item is seg, doc, or set
    assert num_sys * self.num_items == len(gold_scores)
    self.gold_scores = gold_scores
    self.metric_scores = metric_scores
    self.none_count = gold_scores.count(None)

  def GenCorrFunction(self, corr_fcn, averaged, by_system=False,
                      replace_nans_with_zeros=False):
    """Convenience function to create a correlation functor for these stats."""
    return CorrFunction(corr_fcn, self.num_sys if averaged else 0,
                        self.none_count, by_system, replace_nans_with_zeros)

  def Pearson(self, averaged=False, by_system=False,
              replace_nans_with_zeros=False):
    """Pearson correlation and pvalue, optionally averaged over items."""
    cf = self.GenCorrFunction(scipy.stats.pearsonr, averaged, by_system,
                              replace_nans_with_zeros)
    return cf.Corr(self.gold_scores, self.metric_scores)

  def Spearman(self, averaged=False, by_system=False,
               replace_nans_with_zeros=False):
    """Spearman correlation and pvalue, optionally averaged over items."""
    cf = self.GenCorrFunction(scipy.stats.spearmanr, averaged, by_system,
                              replace_nans_with_zeros)
    return cf.Corr(self.gold_scores, self.metric_scores)

  def Kendall(self, averaged=False, by_system=False,
              replace_nans_with_zeros=False):
    """Kendall correlation and pvalue, optionally averaged over items."""
    cf = self.GenCorrFunction(scipy.stats.kendalltau, averaged, by_system,
                              replace_nans_with_zeros)
    return cf.Corr(self.gold_scores, self.metric_scores)

  def KendallLike(self, averaged=True, thresh=25):
    """WMT Kendall-like corr and stats, optionally averaged over items."""
    cf = KendallLike(self.num_sys if averaged else 1, thresh)
    return cf.Corr(self.gold_scores, self.metric_scores)


class CorrFunction:
  """Wrap any correlation from scipy.stats, with optional averaging and Nones.

  This is a functor with the same interface as the scipy.stats functions:
  call with two float vectors, returns a correlation and pvalue. It wraps an
  arbitrary correlation function to provide optional averaging over per-system
  or per-item scores and removal of None entries.
  """

  def __init__(self, corr_fcn, num_sys=0, filter_nones=False, by_system=False,
               replace_nans_with_zeros=False):
    """Construct with correlation function and optional arguments.

    Args:
      corr_fcn: Function that maps two float vectors to a (correlation, pvalue)
        tuple, for instance scipy.stats.pearsonr.
      num_sys: If greater than 0, indicates that the vector arguments to
        corr_fcn contain num_sys blocks of item scores grouped by system. The
        returned correlation and pvalue are averages over per-item
        correlations.  Short per-item lists can lead to repeated scores that
        make correlations undefined; by default these are discarded, and the
        Corr() function returns the number of items that were actually used
        (last value of the returned triple). If num_sys is 0, a single
        correlation will be computed over the input vectors.
      filter_nones: If true, any None values in the first vector argument
        (assumed to represent gold scores) are filtered in tandem from both
        vector arguments before computing correlations.
      by_system: If true, this computes averages over correlations for the
        blocks of scores for each system, rather than for the lists of items at
        the same position within each block. No effect if num_sys is 0.
      replace_nans_with_zeros: If true, replace NaN correlations with 0 rather
        than discarding them as described above.
    """
    self._corr_fcn = corr_fcn
    self._num_sys = num_sys
    self._filter_nones = filter_nones
    self._by_system = by_system and num_sys > 0
    self._replace_nans_with_zeros = replace_nans_with_zeros

  def __call__(self, vect1, vect2):
    return self.Corr(vect1, vect2)[:2]

  def Corr(self, vect1, vect2):
    """Return correlation, pvalue, and number of items used for averaging."""
    # Reshape into item x system score matrices, for average over items.
    num_sys = self._num_sys or len(vect1)
    mat1 = np.asarray(vect1).reshape(num_sys, -1)
    mat2 = np.asarray(vect2).reshape(num_sys, -1)
    if not self._by_system:
      mat1 = mat1.transpose()
      mat2 = mat2.transpose()
    tot_corr, tot_pval, n = 0, 0, 0
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      for r1, r2 in zip(mat1, mat2):
        if self._filter_nones:
          filt = [(v1, v2) for v1, v2 in zip(r1, r2) if v1 is not None]
          if not filt: continue
          r1, r2 = zip(*filt)
        cv, pv = self._corr_fcn(r1, r2)
        if not math.isnan(cv):
          tot_corr += cv
          tot_pval += pv
          n += 1
        elif self._replace_nans_with_zeros:
          n += 1
    return (tot_corr / n, tot_pval / n, n) if n else (0, 0, 0)


class KendallLike:
  """WMT 'Kendall-like' correlation, with optional averaging.

  This is a functor with the same interface as the scipy.stats functions:
  call with two float vectors, returns a correlation and pvalue.

  Averaging requires that you construct with num_sys > 0, indicating that input
  vectors will contain num_sys sets of item scores grouped by system. The
  returned correlation is a micro-average over per-item pairwise comparisons.
  Any None entries in the first vector argument (assumed to represent gold
  scores) do not participate in pairwise comparisons.

  The threshold argument applies to the first vector argument only, and filters
  out score pairs whose absolute difference < thresh.
  """

  def __init__(self, num_sys=0, thresh=25):
    self._num_sys = num_sys
    self._thresh = thresh

  def __call__(self, vect1, vect2):
    return self.Corr(vect1, vect2)[:2]

  def Corr(self, vect1, vect2):
    """Return correlation and stats about number and nature of pairs."""
    num_sys = self._num_sys or len(vect1)
    mat1 = np.asarray(vect1).reshape(num_sys, -1).transpose()
    mat2 = np.asarray(vect2).reshape(num_sys, -1).transpose()
    concordant, discordant = 0, 0
    for m1, m2 in zip(mat1, mat2):
      for (a, b) in itertools.combinations(zip(m1, m2), 2):
        if a[0] is None or b[0] is None:
          continue
        diff = a[0] - b[0]  # difference in gold scores
        if abs(diff) >= self._thresh:
          # Deliberate inconsistency between diff >= thresh and diff > 0 to
          # emulate WMT behaviour.
          if diff > 0 and a[1] > b[1] or diff < 0 and a[1] < b[1]:
            concordant += 1
          else:
            discordant += 1
    num_pairs = concordant + discordant
    corr = (concordant - discordant) / num_pairs if num_pairs else 0
    return corr, num_pairs, concordant, discordant


def Agreement(vect1, vect2):
  """Compute pairwise agreement over two vectors, vect1 assumed to be gold."""
  agree, num_pairs = 0, 0
  for a, b in itertools.combinations(zip(vect1, vect2), 2):
    if a[0] is None or b[0] is None:
      continue
    agree += np.sign(a[0] - b[0]) == np.sign(a[1] - b[1])
    num_pairs += 1
  return agree, num_pairs


class KendallPreproc:
  """Optional preprocessing for `KendallPython`.

  This factors out computation that depends only on one of the two vectors to
  be compared in the `KendallPython` function. It involves a slight overhead
  (due to a less efficient calculation of the number of ties), so should only
  be used when a given vector is to be compared with many other vectors.
  """

  def __init__(self, y):
    """Initialize with a vector of numeric values."""
    y = np.asarray(y)
    self.perm = np.argsort(y)
    y = y[self.perm]
    self.y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)
    _, counts = np.unique(np.asarray(y), return_counts=True)
    self.ytie = sum(n * (n - 1) // 2 for n in counts)


def KendallPython(
    x: ArrayLike,
    y: ArrayLike,
    variant: str = 'b',
    preproc: Optional[KendallPreproc] = None,
) -> Tuple[float, float]:
  """Lightweight, optionally factored version of Kendall's tau.

  This is based on the scipy.stats implementation of Kendall's tau
  https://github.com/scipy/scipy/blob/745bf604640969a25c18f6d6ace166701fac0429/scipy/stats/_stats_py.py#L5474
  with the following changes:
  1) The cython function for computing discordant pairs is replaced by inline
     python. This works up to 2x faster for small vectors (< 50 elements), which
     can be advantageous when processing many such vectors.
  2) The part of the computation that depends solely on y can optionally be
     factored out, for applications involving comparison of multiple x vectors
     to a single y vector. See `KendallPreproc`. This is typically a time
     savings of 10-15%.
  3) The p-value calculation and associated arguments are omitted.
  4) The input vectors are assumed not to contain NaNs.

  Args:
    x: Vector of numeric values.
    y: Vector of numeric values.
    variant: Either 'b' or 'c' to compute the respective tau variant.
    preproc: A `KendallProc` object that has been called on a vector of y values
      to be compared to the currrent x. If this is non-None, the y parameter is
      ignored.

  Returns:
    A tuple (k, 0) where the first element is the Kendall statistic and the
    second is a dummy value for compatibility with `scipy.stat.kendalltau`.
  """

  x = np.asarray(x)
  if not preproc:
    y = np.asarray(y)

  size = x.size
  if preproc is None:
    perm = np.argsort(y)
    x, y = x[perm], y[perm]
    y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)
  else:
    x, y = x[preproc.perm], preproc.y

  # stable sort on x and convert x to dense ranks
  perm = np.argsort(x, kind='mergesort')
  x, y = x[perm], y[perm]
  x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)

  # count discordant pairs
  sup = 1 + np.max(y)
  # Use of `>> 14` improves cache performance of the Fenwick tree (see gh-10108)
  arr = np.zeros(sup + ((sup - 1) >> 14), dtype=np.intp)
  i, k, idx, dis = 0, 0, 0, 0
  while i < x.size:
    while k < x.size and x[i] == x[k]:
      dis += i
      idx = y[k]
      while idx != 0:
        dis -= arr[idx + (idx >> 14)]
        idx = idx & (idx - 1)
      k += 1
    while i < k:
      idx = y[i]
      while idx < sup:
        arr[idx + (idx >> 14)] += 1
        idx += idx & -idx
      i += 1

  obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
  cnt = np.diff(np.nonzero(obs)[0]).astype('int64', copy=False)

  def CountRankTie(ranks):
    cnt = np.bincount(ranks).astype('int64', copy=False)
    cnt = cnt[cnt > 1]
    return int((cnt * (cnt - 1) // 2).sum())

  ntie = int((cnt * (cnt - 1) // 2).sum())  # joint ties
  xtie = CountRankTie(x)  # ties in x
  ytie = CountRankTie(y) if not preproc else preproc.ytie  # ties in y

  tot = (size * (size - 1)) // 2

  if xtie == tot or ytie == tot:
    return np.nan, 0

  con_minus_dis = tot - xtie - ytie + ntie - 2 * dis
  if variant == 'b':
    tau = con_minus_dis / np.sqrt(tot - xtie) / np.sqrt(tot - ytie)
  elif variant == 'c':
    minclasses = min(len(set(x)), len(set(y)))
    tau = 2 * con_minus_dis / (size**2 * (minclasses - 1) / minclasses)
  else:
    raise ValueError(
        f'Unknown variant of the method chosen: {variant}. '
        "variant must be 'b' or 'c'.")

  return tau, 0


def WilliamsSigDiff(corr1, corr2, corr_fcn, one_sided=True):
  """Determine if there is a significant difference between two correlations.

  Use William's test as advocated by https://www.aclweb.org/anthology/D14-1020
  to decide if the correlation for the metric in corr1 is significantly
  greater than that in corr2 (or vice versa, the test is symmetrical), eg the
  returned p-value is < 0.05.

  This function works with arbitrary correlation functions, but the
  interpretation of results other than non-averaged Pearson with complete
  entries is not clear.

  Args:
    corr1: Correlation object for metric1.
    corr2: Correlation object for metric2.
    corr_fcn: Correlation function: maps 2 float vectors to corr, pval. Use the
      CorrFunction or KendallLike functors if gold vectors contain None entries
      or you want averaging.
    one_sided: Use a one-sided test if true (recommended), else two-sided.

  Returns:
    Tuple (pval, correlation1, correlation2).
  """
  if corr1.gold_scores != corr2.gold_scores:
    raise ValueError('Gold scores for correlations don\'t match')
  r1 = corr_fcn(corr1.gold_scores, corr1.metric_scores)[0]
  r2 = corr_fcn(corr2.gold_scores, corr2.metric_scores)[0]
  r12 = corr_fcn(corr1.metric_scores, corr2.metric_scores)[0]
  n = len(corr1.gold_scores)
  return WilliamsTest(r1, r2, r12, n, one_sided), r1, r2


def WilliamsTest(r12, r13, r23, n, one_sided=True):
  """Return Williams test p-value for given Pearson correlations."""
  k = 1 - r12**2 - r13**2 - r23**2 + 2 * r12 * r23 * r13
  rbar = ((r12 + r13) / 2)
  tnum = (r12 - r13) * np.sqrt((n - 1) * (1 + r23))
  tden = np.sqrt(2 * (n - 1) / (n - 3) * k + rbar**2 * (1 - r23)**3)
  p = scipy.stats.t.sf(np.abs(tnum / tden), n - 3)
  return p if one_sided else 2 * p


def _ReshapeAndFilter(
    corr1: Correlation, corr2: Correlation, average_by: str
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
  """Prepare candidate score vectors for paired significance tests.

  Helper function for PermutationSigDiff that creates flattened vectors for
  gold and metric scores, ordered correctly for a given averaging method, and
  filtered to contain no entries where the gold score is missing.

  Args:
    corr1: Correlation stats for metric1.
    corr2: Correlation stats for metric2.
    average_by: Either 'sys', 'item' or 'none' to group by rows, columns, or
      neither.

  Returns:
    Tuple containing:
    - list of sizes for consecutive groups in returned vectors
    - gold scores packed into a 1D numpy array
    - scores from metric1 packed into a 1D numpy array
    - scores from metric2 packed into a 1D numpy array

  """
  if corr1.gold_scores != corr2.gold_scores:
    raise ValueError('Gold scores for correlations don\'t match')

  nrows = 1 if average_by == 'none' else corr1.num_sys
  gold = np.asarray(corr1.gold_scores).reshape(nrows, -1)
  scores1 = np.asarray(corr1.metric_scores).reshape(nrows, -1)
  scores2 = np.asarray(corr2.metric_scores).reshape(nrows, -1)
  if average_by == 'item':
    gold = gold.transpose()
    scores1 = scores1.transpose()
    scores2 = scores2.transpose()

  lens, gold_filt, scores1_filt, scores2_filt = [], [], [], []
  okay = gold != None  # pylint: disable=singleton-comparison
  for ok, g, s1, s2 in zip(okay, gold, scores1, scores2):
    num_elems = ok.sum()
    if num_elems:
      lens.append(num_elems)
      gold_filt.extend(g[ok])
      scores1_filt.extend(s1[ok])
      scores2_filt.extend(s2[ok])

  gold = np.asarray(gold_filt)
  scores1 = np.asarray(scores1_filt)
  scores2 = np.asarray(scores2_filt)

  return lens, gold, scores1, scores2


def PermutationSigDiff(
    corr1: Correlation,
    corr2: Correlation,
    corr_fcn: Callable[[ArrayLike, ArrayLike], tuple[float, float]],
    average_by: str = 'none',
    k: int = 1000,
    block_size: int = 1000,
    early_min: float = 0.02,
    early_max: float = 0.5,
    replace_nans_with_zeros: bool = False,
    fast_kendall: bool = True):
  """Determine if there is a significant difference between two correlations.

  Uses the PERM-BOTH permutation test advocated by
  https://arxiv.org/abs/2104.00054 to decide if the correlation for the metric
  in corr2 is significantly greater than that in corr1. Returns a p-value for
  the hypothesis that metric2 correlates better, or equivalently 1 minus the
  p-value for the hypothesis that metric1 correlates better.

  Args:
    corr1: Statistics for metric1.
    corr2: Statistics for metric2.
    corr_fcn: Correlation function: maps 2 float vectors to corr, pval. This
      should be a plain function like the ones in scipy.stats rather than a
      wrapper.
    average_by: Either 'sys', 'item' or 'none' to group by rows, columns, or
      neither.
    k: Number of resampling runs.
    block_size: Size of blocks for early stopping checks. Set to k for no early
      stopping.
    early_min: Early stop if pval < early_min at current block boundary.
    early_max: Early stop if pval > early_max at current block boundary.
    replace_nans_with_zeros: Replace NaNs with 0, otherwise discard.
    fast_kendall: Use KendallPython instead of scipy.stats.kendall when doing
      item-wise averaging. No effect if corr_fcn is not scipy.stats.kendall.

  Returns:
    - p-value for correlation of metric2 > correlation of metric1
    - delta: corr_fcn(metric2) - corr_fcn(metric1)
    - k_used: number of resampling runs actually performed
  """
  lens, gold, mscores1, mscores2 = _ReshapeAndFilter(corr1, corr2, average_by)
  mscores1 = scipy.stats.zscore(mscores1)
  mscores2 = scipy.stats.zscore(mscores2)
  starts = np.r_[0, np.cumsum(lens)]
  bounds = list(zip(starts[:-1], starts[1:]))

  preprocs = None
  if (corr_fcn == scipy.stats.kendalltau and average_by == 'item'
      and fast_kendall):
    preprocs = [KendallPreproc(gold[b: e]) for b, e in bounds]

  def _Corr(mscores):
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      if preprocs:
        vals = [KendallPython(mscores[b: e], None, 'b', pp)[0]
                for (b, e), pp in zip(bounds, preprocs)]
      else:
        vals = [corr_fcn(gold[b: e], mscores[b: e])[0] for b, e in bounds]
      if replace_nans_with_zeros:
        vals = np.nan_to_num(vals)
      else:
        vals = np.asarray(vals)[~np.isnan(vals)]
      return np.average(vals) if len(vals) else 0

  delta = _Corr(mscores2) - _Corr(mscores1)
  large_delta_count = 0
  for i in range(1, k + 1):
    w1 = np.random.binomial(1, 0.5, len(mscores1))
    w2 = 1 - w1
    m1 = w1 * mscores1 + w2 * mscores2
    m2 = w2 * mscores1 + w1 * mscores2
    if _Corr(m2) - _Corr(m1) >= delta:
      large_delta_count += 1
    if i % block_size == 0:
      pval = large_delta_count / i
      if pval < early_min or pval > early_max:
        break

  return large_delta_count / i, delta, i
