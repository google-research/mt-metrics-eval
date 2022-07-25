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
import warnings
import numpy as np
import scipy.stats


class Correlation:
  """Data and algorithms for computing correlations.

  See dosctrings for CorrFunction and KendallLike for notes on averaging and
  handling gold scores that contain missing entries.
  """

  def __init__(self, num_sys, gold_scores, metric_scores):
    """Construct from parallel vectors that group scores by system."""
    self.num_sys = num_sys
    self.num_items = len(gold_scores) // num_sys  # item is seg, doc, or set
    assert num_sys * self.num_items == len(gold_scores)
    self.gold_scores = gold_scores
    self.metric_scores = metric_scores
    self.none_count = gold_scores.count(None)

  def GenCorrFunction(self, corr_fcn, averaged):
    """Convenience function to create a correlation functor for these stats."""
    return CorrFunction(corr_fcn, self.num_sys if averaged else 0,
                        self.none_count)

  def Pearson(self, averaged=False):
    """Pearson correlation and pvalue, optionally averaged over items."""
    cf = self.GenCorrFunction(scipy.stats.pearsonr, averaged)
    return cf.Corr(self.gold_scores, self.metric_scores)

  def Spearman(self, averaged=False):
    """Spearman correlation and pvalue, optionally averaged over items."""
    cf = self.GenCorrFunction(scipy.stats.spearmanr, averaged)
    return cf.Corr(self.gold_scores, self.metric_scores)

  def Kendall(self, averaged=False):
    """Kendall correlation and pvalue, optionally averaged over items."""
    cf = self.GenCorrFunction(scipy.stats.kendalltau, averaged)
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

  def __init__(self, corr_fcn, num_sys=0, filter_nones=False, by_system=False):
    """Construct with correlation function and optional arguments.

    Args:
      corr_fcn: Function that maps two float vectors to a (correlation, pvalue)
        tuple, for instance scipy.stats.pearsonr.
      num_sys: If greater than 0, indicates that the vector arguments to
        corr_fcn contain num_sys blocks of item scores grouped by system. The
        returned correlation and pvalue are averages over per-item correlations.
        Short per-item lists can lead to repeated scores that make correlations
        undefined; these are discarded, and the Corr() function returns the
        number of items that were actually used (last value of the returned
        triple). If num_sys is 0, a single correlation will be computed over the
        input vectors.
      filter_nones: If true, any None values in the first vector argument
        (assumed to represent gold scores) are filtered in tandem from both
        vector arguments before computing correlations.
      by_system: If true, this computes averages over correlations for the
        blocks of scores for each system, rather than for the lists of items at
        the same position within each block. No effect if num_sys is 0.
    """
    self._corr_fcn = corr_fcn
    self._num_sys = num_sys
    self._filter_nones = filter_nones
    self._by_system = by_system and num_sys > 0

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


def PermutationSigDiff(corr1, corr2, corr_fcn, k=1000):
  """Determine if there is a significant difference between two correlations.

  Uses the PERM-BOTH permutation test advocated by
  https://arxiv.org/abs/2104.00054 to decide if the correlation for the metric
  in corr2 is significantly greater than that in corr1. Returns a p-value for
  the hypothesis that metric2 correlates better, or equivalently 1 minus the
  p-value for the hypothesis that metric1 correlates better.

  Args:
    corr1: Correlation object for metric1.
    corr2: Correlation object for metric2.
    corr_fcn: Correlation function: maps 2 float vectors to corr, pval. Use the
      CorrFunction or KendallLike functors if gold vectors contain None entries
      or you want averaging.
    k: Number of resampling runs.

  Returns:
    P-value for corr2 > corr1.
  """
  if corr1.gold_scores != corr2.gold_scores:
    raise ValueError('Gold scores for correlations don\'t match')
  gscores = corr1.gold_scores
  mscores1 = scipy.stats.zscore(corr1.metric_scores)
  mscores2 = scipy.stats.zscore(corr2.metric_scores)

  delta = corr_fcn(gscores, mscores2)[0] - corr_fcn(gscores, mscores1)[0]
  large_delta_count = 0
  for _ in range(k):
    w1 = np.random.binomial(1, 0.5, len(mscores1))
    w2 = 1 - w1
    m1 = w1 * mscores1 + w2 * mscores2
    m2 = w2 * mscores1 + w1 * mscores2
    if corr_fcn(gscores, m2)[0] - corr_fcn(gscores, m1)[0] >= delta:
      large_delta_count += 1
  return large_delta_count / k
