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
"""Correlations and related functions.

This module provides:

- A wrapper (AverageCorrelation) that applies vector-based correlation functions
  to matrices in different ways, and a convenience class (Correlation) that
  stores gold and metric scores, and facilitates calling AverageCorrelation on
  them.
- Various vector-based correlation functions not implemented in scipy.stats;
  currently these are all variants on Kendall's Tau.
- Statistical tests (Williams and Permutation) for determining whether the
  difference between two correlations is significant.
"""

import dataclasses
import functools
import itertools
import math
from typing import Callable
import warnings
from mt_metrics_eval import pce
from mt_metrics_eval import tau_optimization
import numpy as np
import numpy.typing
import scipy.special
import scipy.stats


ArrayLike = numpy.typing.ArrayLike


class Correlation:
  """Data and algorithms for computing correlations.

  This class stores gold and metric scores, and lets you compute different
  correlation statistics for them. The scores can be at the "system"-level
  (conceptually, a single vector) or at the "segment"-level (conceptuatlly,
  a matrix of system x item scores). Both are stored as vectors. The gold scores
  may also contain None values, which are filtered out, in parallel with the
  corresponding metric scores, before computing correlations.

  All correlation functions take an 'average_by' argument that describes how a
  matrix of segment-level scores gets converted into vectors in order to compute
  correlations: 'none' - flatten the matrix into two long vectors; 'sys' -
  compute correlations over corresponding row vectors, then average; or 'item' -
  compute correlations over corresponding column vectors, then average. For
  system-level scores, 'none' is the only option that makes sense.

  For simplicity, the correlation functions here all take only the 'average_by'
  argument, in addition to arguments that are specific to the actual correlation
  statistic. The AverageCorrelation class supports a few other averaging
  options; use that class directly if you need those.
  """

  def __init__(self, num_sys, gold_scores, metric_scores):
    """Construct from parallel vectors that group scores by system."""
    self.num_sys = num_sys
    self.num_items = len(gold_scores) // num_sys  # item is seg, doc, or set
    assert num_sys * self.num_items == len(gold_scores)
    self.gold_scores = gold_scores
    self.metric_scores = metric_scores
    self.none_count = gold_scores.count(None)

  def AverageCorrelation(
      self,
      corr_fcn,
      average_by,
      replace_nans_with_zeros=False,
      macro=True,
      **corr_fcn_args):
    """Avg correlation for current gold & metric vectors, with given options."""
    return AverageCorrelation(
        corr_fcn, self.num_sys, average_by, self.none_count,
        replace_nans_with_zeros, macro, **corr_fcn_args)

  def Pearson(self, average_by='none'):
    cf = self.AverageCorrelation(scipy.stats.pearsonr, average_by)
    return cf(self.gold_scores, self.metric_scores)

  def Spearman(self, average_by='none'):
    cf = self.AverageCorrelation(scipy.stats.spearmanr, average_by)
    return cf(self.gold_scores, self.metric_scores)

  def Kendall(self, average_by='none', variant='b'):
    cf = self.AverageCorrelation(
        scipy.stats.kendalltau, average_by, variant=variant)
    return cf(self.gold_scores, self.metric_scores)

  def KendallLike(self, average_by='item', thresh=25):
    """WMT Kendall-like corr and stats, averaged over items by default."""
    cf = self.AverageCorrelation(
        KendallLike, average_by, macro=False, thresh=thresh)
    return cf(self.gold_scores, self.metric_scores)

  def KendallVariants(self, average_by='none', variant='b', epsilon=0.0):
    """Kendall Variants, including '23' and 'acc23'."""
    cf = self.AverageCorrelation(
        KendallVariants, average_by, variant=variant, epsilon=epsilon)
    return cf(self.gold_scores, self.metric_scores)

  def KendallWithTiesOpt(
      self, average_by='none', variant='acc23', sample_rate=0.1):
    """Kendall with ties, with optimized threshold."""
    cf = self.AverageCorrelation(
        KendallWithTiesOpt, average_by, variant=variant,
        sample_rate=sample_rate)
    return cf(self.gold_scores, self.metric_scores)

  def PairwiseConfidenceError(self, average_by='none'):
    """System-level soft pairwise accuracy."""
    cf = self.AverageCorrelation(PairwiseConfidenceError, average_by)
    return cf(self.gold_scores, self.metric_scores)


def filter_gold_nones(
    gold: ArrayLike, model: ArrayLike
) -> tuple[ArrayLike, ArrayLike]:
  """Filters pairs where the gold score is None.

  If gold[i] is None, gold[i] and model[i] will be removed and not returned in
  the output.

  Args:
    gold: The gold scores.
    model: The model scores.

  Returns:
    The gold and model scores where the gold score is not None.
  """
  filt = [(v1, v2) for v1, v2 in zip(gold, model) if v1 is not None]
  if not filt:
    return [], []
  return zip(*filt)


class AverageCorrelation:
  """Wrap a correlation function to provide averaging and None filtering."""

  # pylint: disable=g-bare-generic
  def __init__(
      self,
      corr_fcn: Callable[..., tuple],
      num_sys: int = 0,
      average_by: str = 'none',
      filter_nones: bool = True,
      replace_nans_with_zeros: bool = False,
      macro: bool = True,
      **corr_fcn_args):
    """Construct with a correlation function and optional arguments.

    Args:
      corr_fcn: Function that maps two float vectors, followed by optional
        additional arguments, to a correlation, pvalue pair, followed by
        optional additional elements, for instance scipy.stats.pearsonr.
        KendallWithTiesOpt is treated as a special case, since it performs its
        own averaging.
      num_sys: Indicates that the vectors passed to Corr() are packed matrices
        containing num_sys consecutive rows. Ignored if average_by is 'none'.
      average_by: The averaging to be performed by Corr(), one of 'sys', 'item',
        or 'none' to average correlations over corresponding matrix rows or
        columns, or to just compute a single correlation over the input vectors.
      filter_nones: If True, any None values in the first vector argument
        (assumed to represent gold scores) are filtered in tandem from both
        vector arguments before computing correlations. Set this to False for
        more efficient operation if you know there are no None's in the input.
      replace_nans_with_zeros: When averaging, correlations that are NaN (eg,
        due to short vectors or repeated values) are normally removed before 
        computing the average. This option replaces them with 0 instead.
      macro: If True, Corr() returns a plain average over row- or item-wise
        correlations. If False, then if corr_fcn returns three values, the 3rd
        is used to weight each sys- or item-wise correlation, otherise the
        number of non-None scores in each row or column is used as a weight.
      **corr_fcn_args: Optional extra arguments to corr_fcn.
    """
    self._corr_fcn = corr_fcn
    self._num_sys = num_sys
    self._average_by = average_by
    self._filter_nones = filter_nones
    self._replace_nans_with_zeros = replace_nans_with_zeros
    self._macro = macro
    self._corr_fcn_args = corr_fcn_args

  def __call__(self, gold_vect, metric_vect):
    """Same as standard correlation function, with one extra return value."""
    return self.Corr(gold_vect, metric_vect)

  def Corr(self, gold_vect: ArrayLike, metric_vect: ArrayLike):
    """Return correlation, pvalue, and denominator used for averaging."""

    mat1 = _Reshape(gold_vect, self._num_sys, self._average_by)
    mat2 = _Reshape(metric_vect, self._num_sys, self._average_by)

    # KendallWithTiesOpt does its own averaging.
    if self._corr_fcn is KendallWithTiesOpt:
      corr, _, _ = KendallWithTiesOpt(
          gold_vect, metric_vect, num_sys=self._num_sys,
          average_by=self._average_by, **self._corr_fcn_args)
      return corr, 0, mat1.shape[0]
    # PairwiseConfidenceError does its own averaging.
    elif self._corr_fcn is PairwiseConfidenceError:
      corr = PairwiseConfidenceError(
          gold_vect, metric_vect, num_sys=self._num_sys,
          filter_nones=self._filter_nones, **self._corr_fcn_args)
      return corr[0], 0, mat1.shape[0]

    tot_corr, tot_pval, n, k = 0, 0, 0, 1
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      for r1, r2 in zip(mat1, mat2):
        if self._filter_nones:
          r1, r2 = filter_gold_nones(r1, r2)
          if not r1 or len(r1) == 1: continue
        ret = self._corr_fcn(r1, r2, **self._corr_fcn_args)
        if not self._macro:
          k = ret[2] if len(ret) > 2 else len(r1)
        if not math.isnan(ret[0]):
          tot_corr += k * ret[0]
          tot_pval += k * ret[1]
          n += k
        elif self._replace_nans_with_zeros:
          n += k
    return (tot_corr / n, tot_pval / n, n) if n else (0, 0, 0)


def KendallLike(
    gold_scores: ArrayLike,
    metric_scores: ArrayLike,
    thresh: float = 25.0,
) -> tuple[float, float, int, int, int]:
  """WMT20 'Kendall-like' correlation between two vectors."""
  concordant, discordant = 0, 0
  for (a, b) in itertools.combinations(zip(gold_scores, metric_scores), 2):
    if a[0] is None or b[0] is None:
      continue
    diff = a[0] - b[0]  # difference in gold scores
    if abs(diff) >= thresh:
      # Deliberate inconsistency between diff >= thresh and diff > 0 to
      # emulate WMT behaviour.
      if diff > 0 and a[1] > b[1] or diff < 0 and a[1] < b[1]:
        concordant += 1
      else:
        discordant += 1
  num_pairs = concordant + discordant
  corr = (concordant - discordant) / num_pairs if num_pairs else 0
  return corr, 0, num_pairs, concordant, discordant


def Agreement(gold_vect, metric_vect):
  """Pairwise agreement over gold and metric vectors."""
  agree, num_pairs = 0, 0
  for a, b in itertools.combinations(zip(gold_vect, metric_vect), 2):
    if a[0] is None or b[0] is None:
      continue
    agree += np.sign(a[0] - b[0]) == np.sign(a[1] - b[1])
    num_pairs += 1
  return agree, num_pairs


class KendallPreproc:
  """Optional preprocessing for `KendallVariants`.

  This factors out computation that depends only on one of the two vectors to
  be compared in the `KendallVariants` function. It involves a slight overhead
  (due to a less efficient calculation of the number of ties and because it
  calculates preprocessed data for both the Fenwick tree and matrix
  implementations of `KendallVariants`), so should only be used when a given
  vector is to be compared with many other vectors.
  """

  def __init__(self, y):
    """Initialize with a vector of numeric values."""
    self.y = np.asarray(y)

    # Matrix preprocessing
    y1, y2 = np.meshgrid(self.y, self.y.T)
    self.y_diffs = y1 - y2
    self.y_is_tie = self.y_diffs == 0.0

    # Fenwick tree preprocessing
    self.perm = np.argsort(self.y)
    y_sorted = self.y[self.perm]
    self.y_cumsum = np.r_[True, y_sorted[1:] != y_sorted[:-1]].cumsum(
        dtype=np.intp
    )
    _, counts = np.unique(np.asarray(y_sorted), return_counts=True)
    self.ytie = sum(n * (n - 1) // 2 for n in counts)


class PairwiseDiffs:
  """Matrix of pairwise score differences over a vector of values."""

  def __init__(self, x=None, epsilon=0.0):
    if x is not None:
      x = np.asarray(x)
      x1, x2 = np.meshgrid(x, x.T)
      self.x_diffs = x1 - x2
      # Introduce ties into x by setting the diffs to 0 if they are <= epsilon
      self.x_is_tie = np.abs(self.x_diffs) <= epsilon
      self.x_diffs[self.x_is_tie] = 0.0

  def Combine(self, other, self_wts, other_wts):
    """Combine with binary weight matrices."""
    ret = PairwiseDiffs()
    ret.x_diffs = self_wts  * self.x_diffs + other_wts * other.x_diffs
    ret.x_is_tie = self_wts * self.x_is_tie + other_wts * other.x_is_tie
    return ret


def _MatrixSufficientStatistics(
    x: ArrayLike,
    y: ArrayLike,
    epsilon: float,
    preproc: KendallPreproc | None,
    preproc_x: KendallPreproc | None,
    ) -> tuple[int, int, int, int, int]:
  """Calculates tau sufficient statistics using matrices in NumPy.

  An absolute difference less than `epsilon` in x pairs is considered to be
  a tie.

  Args:
    x: Vector of numeric values.
    y: Vector of numeric values.
    epsilon: The threshold for which an absolute difference in x scores should
      be considered a tie.
    preproc: A `KendallPreproc` object that has been called on a vector of
      y values to be compared to the currrent x. If this is non-None, the y
      parameter is ignored.
    preproc_x: A `PairwiseDiffs` object that has been called on a vector of
      x values to be compared to the currrent y. If this is non-None, the x
      parameter is ignored.

  Returns:
    The number of concordant pairs, discordant pairs, pairs tied only in x,
    paired tied only in y, and pairs tied in both x and y.
  """
  if preproc_x is None:
    preproc_x = PairwiseDiffs(x, epsilon)
  x_diffs = preproc_x.x_diffs
  x_is_tie = preproc_x.x_is_tie

  if preproc is None:
    y1, y2 = np.meshgrid(y, y.T)
    y_diffs = y1 - y2
    y_is_tie = y_diffs == 0.0
  else:
    y_diffs = preproc.y_diffs
    y_is_tie = preproc.y_is_tie

  n = len(y) if preproc is None else len(preproc.y)
  num_pairs = int(scipy.special.comb(n, 2))
  # All of the counts are divided by 2 because each pair is double counted. The
  # double counted data will always be an even number, so dividing by 2 will
  # be an integer.
  con = int(
      ((x_diffs > 0) & (y_diffs > 0) | (x_diffs < 0) & (y_diffs < 0)).sum() / 2
  )
  t_x = int((x_is_tie & ~y_is_tie).sum() / 2)
  t_y = int((~x_is_tie & y_is_tie).sum() / 2)
  t_xy = int(((x_is_tie & y_is_tie).sum() - n) / 2)  # -n removes diagonal
  dis = num_pairs - (con + t_x + t_y + t_xy)
  return con, dis, t_x, t_y, t_xy


def _FenwickTreeSufficientStatistics(
    x: ArrayLike,
    y: ArrayLike,
    preproc: KendallPreproc | None,
) -> tuple[int, int, int, int, int, int]:
  """Calculates tau sufficient statistics using the Fenwick Tree method.

  This is based on the scipy.stats implementation of Kendall's tau
  https://github.com/scipy/scipy/blob/745bf604640969a25c18f6d6ace166701fac0429/scipy/stats/_stats_py.py#L5474
  with the following changes:
  1) The cython function for computing discordant pairs is replaced by inline
     python. This works up to 2x faster for small vectors (< 50 elements), which
     can be advantageous when processing many such vectors.
  2) The function returns tau sufficient statistics required to compute any
     tau, not just tau-b/c.

  Args:
    x: Vector of numeric values.
    y: Vector of numeric values.
    preproc: A `KendallPreproc` object that has been called on a vector of y
      values to be compared to the currrent x. If this is non-None, the y
      parameter is ignored.

  Returns:
    The number of concordant pairs, discordant pairs, pairs tied only in x,
    pairs tied only in y, and pairs tied in both x and y.
  """
  size = x.size
  if preproc is None:
    perm = np.argsort(y)
    x, y = x[perm], y[perm]
    y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)
  else:
    x, y = x[preproc.perm], preproc.y_cumsum

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

  def _CountRankTie(ranks):
    cnt = np.bincount(ranks).astype('int64', copy=False)
    cnt = cnt[cnt > 1]
    return int((cnt * (cnt - 1) // 2).sum())

  ntie = int((cnt * (cnt - 1) // 2).sum())  # joint ties
  xtie = _CountRankTie(x)  # ties in x
  ytie = _CountRankTie(y) if not preproc else preproc.ytie  # ties in y

  tot = (size * (size - 1)) // 2
  xtie_only = xtie - ntie
  ytie_only = ytie - ntie
  con = tot - xtie_only - ytie_only - ntie - dis
  return con, dis, xtie_only, ytie_only, ntie


def KendallVariants(
    gold_scores: ArrayLike,
    metric_scores: ArrayLike,
    variant: str = 'b',
    preproc: KendallPreproc | None = None,
    epsilon: float = 0.0,
    metric_preproc: PairwiseDiffs | None = None,
) -> tuple[float, float]:
  """Lightweight, optionally factored versions of variants on Kendall's Tau.

  This function calculates the sufficient statistics for tau in two different
  ways, either using a Fenwick Tree (`_FenwickTreeSufficientStatistics`) when
  `epsilon` is 0 or NumPy matrices (`_MatrixSufficientStatistics`) otherwise.
  Note that the latter implementation has an O(n^2) space requirement, which
  can be significant for long vectors.

  This implementation makes several changes to the SciPy implementation of
  Kendall's tau:
  1) For the Fenwick tree version, the cython function for computing discordant
     pairs is replaced by inline python. This works up to 2x faster for small
     vectors (< 50 elements), which can be advantageous when processing many
     such vectors.
  2) The part of the computation that depends solely on gold_scores can
     optionally be factored out, for applications involving comparison of
     multiple metric vectors to a single gold vector. See `KendallPreproc`.
  3) The p-value calculation and associated arguments are omitted.
  4) The input vectors are assumed not to contain NaNs.

  Args:
    gold_scores: Vector of numeric values.
    metric_scores: Vector of numeric values.
    variant: Either 'b', 'c', '23', or 'acc23' to compute the respective tau
      variant. See https://arxiv.org/abs/2305.14324 for details about the
      '23' and 'acc23' variants.
    preproc: A preprocessing object that has been called on a vector of gold
      values to be compared to the currrent metric_scores. If this is non-None,
      the gold_scores parameter is ignored.
    epsilon: The threshold for which an absolute difference in metric scores
      should be considered a tie.
    metric_preproc: A preprocessing object that has been called on a vector of
      metric values to be compared to the currrent metric_scores. If this is
      non-None, the metric_scores parameter is ignored (unless the variant is
      'c', in which case it must always be explicitly provided).

  Returns:
    A tuple (k, 0) where the first element is the Kendall statistic and the
    second is a dummy value for compatibility with `scipy.stats.kendalltau`.
  """
  if epsilon < 0:
    raise ValueError('Epsilon must be non-negative.')
  if epsilon > 0 and variant == 'c':
    # It's not clear how to define minclasses with a non-zero epsilon.
    raise ValueError('Non-zero epsilon with tau-c not supported.')
  if metric_scores is None and variant == 'c':
    raise ValueError('Metric scores must be provided for tau-c.')

  # The helper functions and tau_optimization expect metric_scores first, the
  # reverse of the convention used for public methods in this module.
  x, y = metric_scores, gold_scores

  x = np.asarray(x) if x is not None else None
  y = np.asarray(y) if y is not None else preproc.y

  if epsilon > 0 or metric_preproc is not None:
    con, dis, xtie_only, ytie_only, tie_both = _MatrixSufficientStatistics(
        x, y, epsilon, preproc, metric_preproc
    )
  else:
    con, dis, xtie_only, ytie_only, tie_both = _FenwickTreeSufficientStatistics(
        x, y, preproc
    )

  size = y.size
  xtie = xtie_only + tie_both
  ytie = ytie_only + tie_both
  tot = con + dis + xtie_only + ytie_only + tie_both

  if variant in ['b', 'c'] and (xtie == tot or ytie == tot):
    return np.nan, 0

  if variant == 'b':
    tau = (con - dis) / np.sqrt(tot - xtie) / np.sqrt(tot - ytie)
  elif variant == 'c':
    minclasses = min(len(set(x)), len(set(y)))
    tau = 2 * (con - dis) / (size**2 * (minclasses - 1) / minclasses)
  elif variant == '23':
    tau = (con + tie_both - dis - xtie_only - ytie_only) / tot
  elif variant == 'acc23':
    tau = (con + tie_both) / tot
  else:
    raise ValueError(
        f'Unknown variant of the method chosen: {variant}. '
        "variant must be 'b', 'c', '23', or 'acc23'.")

  return tau, 0


def KendallWithTiesOpt(
    gold_scores: ArrayLike,
    metric_scores: ArrayLike,
    variant: str = 'acc23',
    num_sys: int = 1,
    average_by: str = 'none',
    sample_rate: float = 0.1,
    ) -> tuple[float, float, tau_optimization.TauOptimizationResult]:
  """Compute optimized Kendall's variants that take ties into account.

  Uses tau_optimization to optimize a tie threshold on the current input, and
  returns the corresponding correlation.

  Note: this function performs its own averaging and None filtering. It
  interprets its input in the same way as AverageCorrelation does.

  Args:
    gold_scores: Vector of numeric values.
    metric_scores: Vector of numeric values.
    variant: Kendall's variant to be optimized, one of '23' or 'acc23'.
    num_sys: Indicates that the input vectors are packed matrices containing
      num_sys consecutive rows. Ignored if average_by is 'none'.
    average_by: The averaging to be performed, one of 'sys', 'item',
      or 'none' to average correlations over corresponding matrix rows or
      columns, or to just compute a single correlation over the input vectors.
    sample_rate: Sample rate to pass to tau_optimization.

  Returns:
    Correlation value, optimal threshold, full optimization result.
  """
  if variant == '23':
    tau_fn = tau_optimization.TauSufficientStats.tau_23
  elif variant == 'acc23':
    tau_fn = tau_optimization.TauSufficientStats.acc_23
  else:
    raise ValueError('Only the *23 variants support epsilon optimization.')

  gold = _Reshape(gold_scores, num_sys, average_by)
  metric = _Reshape(metric_scores, num_sys, average_by)
  # tau_optimization uses reverse convention for gold, metric scores
  opt_result = tau_optimization.tau_optimization(
      metric, gold, tau_fn, sample_rate)
  return opt_result.best_tau, opt_result.best_threshold, opt_result


def PairwiseConfidenceError(
    gold_scores: list[ArrayLike],
    metric_scores: list[ArrayLike],
    num_sys: int,
    num_permutations: int = 1000,
    filter_nones: bool = False,
) -> tuple[float, ...]:
  """Calculates pairwise confidence error (PCE)."""
  if filter_nones:
    gold_scores, metric_scores = filter_gold_nones(
        gold_scores, metric_scores
    )

  # Convert the gold and metric scores into N x M matrices where N is the
  # number of systems and M is the number of segments.
  gold = _Reshape(gold_scores, num_sys, 'sys')
  metric = _Reshape(metric_scores, num_sys, 'sys')

  gold_pvalues = pce.compute_pairwise_p_values(
      gold, num_permutations=num_permutations
  )
  metric_pvalues = pce.compute_pairwise_p_values(
      metric, num_permutations=num_permutations
  )
  return (pce.compute_one_minus_pce(gold_pvalues, metric_pvalues),)


def _Reshape(vector, num_sys, average_by):
  """Reshape a packed vector into a matrix for row averaging."""
  if average_by == 'none':
    return np.asarray(vector).reshape(1, -1)
  elif average_by == 'sys':
    return np.asarray(vector).reshape(num_sys, -1)
  elif average_by == 'item':
    return np.asarray(vector).reshape(num_sys, -1).transpose()
  else:
    raise ValueError(f'Unknown averaging option: {average_by}')


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

  gold = _Reshape(corr1.gold_scores, corr1.num_sys, average_by)
  scores1 = _Reshape(corr1.metric_scores, corr1.num_sys, average_by)
  scores2 = _Reshape(corr2.metric_scores, corr2.num_sys, average_by)

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


@dataclasses.dataclass
class PermutationSigDiffParams:
  """Minor parameters for PermutationSigDiff."""
  # Size of blocks for early stopping checks. Set to k for no early stopping.
  block_size: int = 1000
  # Early stop if pval < early_min at current block boundary
  early_min: float = 0.02
  # Early stop if pval > early_max at current block boundary
  early_max: float = 0.5


# pylint: disable=g-bare-generic
def PermutationSigDiff(
    corr1: Correlation,
    corr2: Correlation,
    corr_fcn: Callable[..., tuple],
    average_by: str = 'none',
    k: int = 1000,
    params: PermutationSigDiffParams = PermutationSigDiffParams(),
    replace_nans_with_zeros: bool = False,
    **corr_fcn_args
    ) -> tuple[float, float, int, list[tuple[float, float]]]:
  """Determine if there is a significant difference between two correlations.

  Uses the PERM-BOTH permutation test advocated by
  https://arxiv.org/abs/2104.00054 to decide if the correlation for the metric
  in corr2 is significantly greater than that in corr1. Returns a p-value for
  the hypothesis that metric2 correlates better, or equivalently 1 minus the
  p-value for the hypothesis that metric1 correlates better.

  Args:
    corr1: Statistics for metric1.
    corr2: Statistics for metric2.
    corr_fcn: Function that maps two float vectors, followed by optional
      additional arguments, to a correlation value, followed by
      optional additional elements. KendallVariants and KendallWithTiesOpt
      trigger special behaviour.
    average_by: Either 'sys', 'item' or 'none' to group by rows, columns, or
      neither.
    k: Number of resampling runs.
    params: Additional minor parameters, see PermutationSigDiffParams.
    replace_nans_with_zeros: When averaging, replace NaNs with 0 rather than
      removing them from the average. No-op if corr_fcn is KendallWithTiesOpt.
    **corr_fcn_args: Optional extra arguments to corr_fcn.

  Returns:
    - p-value for correlation of metric2 > correlation of metric1
    - delta: corr_fcn(metric2) - corr_fcn(metric1)
    - k_used: number of resampling runs actually performed
    - list of resampled (metric2, metric1) correlation pairs
  """
  lens, gold, mscores1, mscores2 = _ReshapeAndFilter(corr1, corr2, average_by)
  mscores1 = scipy.stats.zscore(mscores1)
  mscores2 = scipy.stats.zscore(mscores2)
  starts = np.r_[0, np.cumsum(lens)]
  bounds = list(zip(starts[:-1], starts[1:]))

  preprocs = None
  if corr_fcn is KendallVariants:
    preprocs = [KendallPreproc(gold[b: e]) for b, e in bounds]
  elif corr_fcn is KendallWithTiesOpt:
    gold = corr1.gold_scores
    mscores1 = scipy.stats.zscore(corr1.metric_scores)
    mscores2 = scipy.stats.zscore(corr2.metric_scores)

  def _Corr(mscores):
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      if corr_fcn is KendallWithTiesOpt:
        return KendallWithTiesOpt(
            gold, mscores, num_sys=corr1.num_sys, average_by=average_by,
            **corr_fcn_args)[0]
      elif corr_fcn is KendallVariants:
        vals = [
            KendallVariants(None, mscores[b: e], preproc=pp, **corr_fcn_args)[0]
            for (b, e), pp in zip(bounds, preprocs)]
      else:
        vals = [corr_fcn(gold[b: e], mscores[b: e], **corr_fcn_args)[0]
                for b, e in bounds]
      if replace_nans_with_zeros:
        vals = np.nan_to_num(vals)
      else:
        vals = np.asarray(vals)[~np.isnan(vals)]
      return np.average(vals) if len(vals) else 0

  corrs = []
  delta = _Corr(mscores2) - _Corr(mscores1)
  i, large_delta_count = 1, 0
  for i in range(1, k + 1):
    w1 = np.random.binomial(1, 0.5, len(mscores1))
    w2 = 1 - w1
    m1 = w1 * mscores1 + w2 * mscores2
    m2 = w2 * mscores1 + w1 * mscores2
    c1, c2 = _Corr(m1), _Corr(m2)
    corrs.append((c2, c1))
    if c2 - c1 >= delta:
      large_delta_count += 1
    if i % params.block_size == 0:
      pval = large_delta_count / i
      if pval < params.early_min or pval > params.early_max:
        break

  return large_delta_count / i, delta, i, corrs


def PairwisePermutationSigDiff(
    corr1: Correlation,
    corr2: Correlation,
    variant: str = 'acc23',
    average_by: str = 'none',
    k: int = 1000,
    params: PermutationSigDiffParams = PermutationSigDiffParams(),
    epsilon1: float | None = None,
    epsilon2: float | None = None,
    sample_rate: float = 1.0,
    replace_nans_with_zeros: bool = False,
    ) -> tuple[float, float, int]:
  """Determine if there is a significant delta between two Kendall correlations.

  Perform a permutation test over Kendall correlations by swapping the results
  of pairwise comparisons rather than metric scores. This allows two metrics
  with different optimal tie thresholds to be compared on an equal footing
  without having to perform expensive re-calibration for each random draw in the
  permutation test, as is done in PermutationSigDiff with corr_fcn=
  KendallWithTiesOpt. This is not an approximation of the stanard permutation
  test, since it can generate correlation values that can't result from any
  combination of the two original metric score vectors.

  Args:
    corr1: Statistics for metric1.
    corr2: Statistics for metric2.
    variant: Kendall variant, see KendallVariants.
    average_by: Either 'sys', 'item' or 'none' to group by rows, columns, or
      neither.
    k: Number of resampling runs.
    params: Early-stopping parameters, see PermutationSigDiffParams.
    epsilon1: KendallVariants epsilon argument for metric1. If None, an optimal
      value will be used. This is equivalent to using KendallWithTiesOpt as the
      correlation; it only supports the '*23' variants.
    epsilon2: KendallVariants epsilon argument for metric2. If None, an optimal
      value will be used. This is equivalent to using KendallWithTiesOpt as the
      correlation; it only supports the '*23' variants.
    sample_rate: Sample rate to use for epsilon calibration, see
      KendallWithTiesOpt.
    replace_nans_with_zeros: When averaging, replace NaNs with 0 rather than
      removing them from the average. No-op if corr_fcn is KendallWithTiesOpt.

  Returns:
    - p-value for correlation of metric2 > correlation of metric1
    - delta: corr_fcn(metric2) - corr_fcn(metric1)
    - k_used: number of resampling runs actually performed
    - list of resampled (metric2, metric1) correlation pairs
  """
  if variant == 'c':
    # tau-c depends on the actual metric scores, rather than just the pairwise
    # ranks, so it's not defined once we start permuting pairs.
    raise ValueError('tau-c not supported by PairwisePermutationSigDiff.')
  if epsilon1 is None:
    _, epsilon1, _ = KendallWithTiesOpt(
        corr1.gold_scores, corr1.metric_scores, variant, corr1.num_sys,
        average_by, sample_rate)
  if epsilon2 is None:
    _, epsilon2, _ = KendallWithTiesOpt(
        corr2.gold_scores, corr2.metric_scores, variant, corr2.num_sys,
        average_by, sample_rate)
  if epsilon1 < 0 or epsilon2 < 0:
    raise ValueError('Epsilon must be non-negative.')

  lens, gold, mscores1, mscores2 = _ReshapeAndFilter(corr1, corr2, average_by)
  starts = np.r_[0, np.cumsum(lens)]
  bounds = list(zip(starts[:-1], starts[1:]))

  preprocs_gold, preprocs1, preprocs2, = [], [], []
  for (b, e) in bounds:
    preprocs_gold.append(KendallPreproc(gold[b: e]))
    preprocs1.append(PairwiseDiffs(mscores1[b: e], epsilon1))
    preprocs2.append(PairwiseDiffs(mscores2[b: e], epsilon2))

  def _Corr(preprocs):
    vals = [KendallVariants(None, None, variant, g, 0, m)[0]
            for g, m in zip(preprocs_gold, preprocs)]
    if replace_nans_with_zeros:
      vals = np.nan_to_num(vals)
    else:
      vals = np.asarray(vals)[~np.isnan(vals)]
    return np.average(vals) if len(vals) else 0

  corrs = []
  delta = _Corr(preprocs2) - _Corr(preprocs1)
  i, large_delta_count = 1, 0
  for i in range(1, k + 1):
    # TODO(fosterg): Vectorize if this is too slow.
    hybrids1, hybrids2 = [], []
    for n, m1, m2 in zip(lens, preprocs1, preprocs2):
      w1 = np.triu(np.random.binomial(1, 0.5, (n, n)))
      w1 = w1 + w1.T  # symmetrize, ignoring diagonal since diffs are 0 there
      w2 = 1 - w1
      hybrids1.append(m1.Combine(m2, w1, w2))
      hybrids2.append(m2.Combine(m1, w1, w2))

    c1, c2 = _Corr(hybrids1), _Corr(hybrids2)
    corrs.append((c2, c1))
    if c2 - c1 >= delta:
      large_delta_count += 1
      if i % params.block_size == 0:
        pval = large_delta_count / i
        if pval < params.early_min or pval > params.early_max:
          break
  return large_delta_count / i, delta, i, corrs


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
      AvergeCorrelation wrapper if gold vectors contain None entries or you want
      averaging.
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


class Sample:
  """Sample indexes using a given sampling method."""

  def _SubsampleSizes(self, proportions, size, bin_sizes):
    """Partition `size` elements according to given proportions."""

    # Shuffle to avoid bias when tie breaking.
    shuffled_ind = np.random.permutation(len(proportions))
    proportions = proportions[shuffled_ind]
    bin_sizes = bin_sizes[shuffled_ind]

    ideal_sizes = proportions * size
    sizes = ideal_sizes.astype(int)
    residuals = sizes - ideal_sizes  # Sort largest first.
    residual_indices = np.argpartition(residuals, size - sum(sizes))
    missing = size - sum(sizes)
    sizes[residual_indices[:missing]] += 1

    # Ensure constraints are satisfied by recursively redistributing mass.
    if (bin_sizes < sizes).any():
      assert sum(bin_sizes) >= size
      realloc_indices = np.flatnonzero(sizes <= bin_sizes)
      mass_to_reallocate = sum(np.maximum(sizes - bin_sizes, 0))
      mass_to_reallocate += sum(sizes[realloc_indices])
      sizes = np.minimum(sizes, bin_sizes)
      proportions_to_reallocate = proportions[realloc_indices]
      proportions_to_reallocate /= sum(proportions_to_reallocate)
      bin_sizes_to_reallocate = bin_sizes[realloc_indices]
      realloc_sizes = self._subsample_sizes(
          proportions_to_reallocate, mass_to_reallocate,
          bin_sizes_to_reallocate)
      sizes[realloc_indices] = realloc_sizes
      assert (sizes <= bin_sizes).all()

    assert sum(sizes) == size, (sum(sizes), size)
    return sizes[np.argsort(shuffled_ind)]  # unshuffle

  def __init__(self, total_size, size, method, bin_sizes=None, seed=None):
    """Create sample: draw `size` indexes in [0, total_size)."""

    self.total_size = total_size
    self.size = size
    self.method = method
    self.bin_sizes = bin_sizes
    self.seed = seed

    rng = np.random.default_rng(seed) if seed else np.random.default_rng()
    if size == 0 or size > total_size:
      size = total_size
    if method == 'uniform':
      self.sample = rng.choice(total_size, size, replace=False)
    elif method == 'stratify':
      assert sum(bin_sizes) == total_size
      bin_sizes = np.asarray(bin_sizes)
      proportions = bin_sizes / total_size
      size_per_bin = self._SubsampleSizes(proportions, size, bin_sizes)
      self.sample = []
      beg = 0
      for n, s in zip(bin_sizes, size_per_bin):
        self.sample.extend(rng.choice(n, s, replace=False) + beg)
        beg += n
    else:
      raise ValueError(f'Unknown method: {method}')

  def Select(self, v):
    """Extract selected items from vector of numeric values."""
    assert len(v) == self.total_size
    return np.asarray(v)[self.sample]
