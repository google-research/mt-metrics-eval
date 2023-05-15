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
"""Tests for stats."""

from mt_metrics_eval import stats
import scipy.stats
import unittest


class CorrelationTest(unittest.TestCase):

  def testCorrelation(self):
    corr = stats.Correlation(2, [1, 2, 3, 4], [2, 1, 4, 3])

    self.assertEqual(corr.num_items, 2)
    self.assertEqual(corr.none_count, 0)

    self.assertAlmostEqual(corr.Pearson()[0], 0.6, places=3)
    self.assertAlmostEqual(corr.Pearson(averaged=True)[0], 1.0, places=3)

    self.assertAlmostEqual(corr.Spearman()[0], 0.6, places=3)
    self.assertAlmostEqual(corr.Spearman(averaged=True)[0], 1.0, places=3)

    self.assertAlmostEqual(corr.Kendall()[0], 0.333, places=3)
    self.assertAlmostEqual(corr.Kendall(averaged=True)[0], 1.0, places=3)

    corr = stats.Correlation(3, [1, None, 3, 4, 5, 6], [2, 1, 4, 3, 5, 6])

    self.assertAlmostEqual(corr.Pearson()[0], 0.904, places=3)
    self.assertAlmostEqual(corr.Pearson(averaged=True)[0], 0.991, places=3)

  def testCorrFunction(self):

    # Default behavior is identical to scipy.stats.
    v1, v2 = [1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 5, 6]
    cf = stats.CorrFunction(scipy.stats.pearsonr)
    self.assertEqual(cf(v1, v2), scipy.stats.pearsonr(v1, v2))

    # Default by_system behavior is identical to scipy.stats.
    v1, v2 = [1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 5, 6]
    cf = stats.CorrFunction(scipy.stats.pearsonr, by_system=True)
    self.assertEqual(cf(v1, v2), scipy.stats.pearsonr(v1, v2))

   # Missing entries.
    cf = stats.CorrFunction(scipy.stats.pearsonr, filter_nones=True)
    v1, v2 = [1, None, 3, 4, 5, 6], [2, 1, 4, 3, 5, 6]
    e1, e2 = [1, 3, 4, 5, 6], [2, 4, 3, 5, 6]
    self.assertAlmostEqual(cf(v1, v2), scipy.stats.pearsonr(e1, e2))

    # Averaging.
    v1, v2 = [1, 2, 3, 4, 5, 6], [2, 1, 8, 3, 5, 6]
    cf = stats.CorrFunction(scipy.stats.pearsonr, num_sys=3)
    p1 = scipy.stats.pearsonr([1, 3, 5], [2, 8, 5])
    p2 = scipy.stats.pearsonr([2, 4, 6], [1, 3, 6])
    c12 = cf(v1, v2)
    self.assertAlmostEqual(c12[0], (p1[0] + p2[0]) / 2, places=3)
    self.assertAlmostEqual(c12[1], (p1[1] + p2[1]) / 2, places=3)

    # Averaging by system.
    v1, v2 = [1, 2, 3, 4, 5, 6], [2, 1, 8, 3, 5, 6]
    cf = stats.CorrFunction(
        scipy.stats.pearsonr, num_sys=2, by_system=True)
    p1 = scipy.stats.pearsonr([1, 2, 3], [2, 1, 8])
    p2 = scipy.stats.pearsonr([4, 5, 6], [3, 5, 6])
    c12 = cf(v1, v2)
    self.assertAlmostEqual(c12[0], (p1[0] + p2[0]) / 2, places=3)
    self.assertAlmostEqual(c12[1], (p1[1] + p2[1]) / 2, places=3)

    # Averaging with missing entries.
    v1, v2 = [1, 2, 3, None, 5, 6], [2, 1, 8, 3, 5, 6]
    cf = stats.CorrFunction(scipy.stats.pearsonr, num_sys=3, filter_nones=True)
    p1 = scipy.stats.pearsonr([1, 3, 5], [2, 8, 5])
    p2 = scipy.stats.pearsonr([2, 6], [1, 6])
    c12 = cf(v1, v2)
    self.assertAlmostEqual(c12[0], (p1[0] + p2[0]) / 2, places=3)
    self.assertAlmostEqual(c12[1], (p1[1] + p2[1]) / 2, places=3)

    # Handling NaNs
    v1, v2 = [1, 2, 3, 4, 5, 6], [2, 2, 2, 4, 5, 6]
    cf = stats.CorrFunction(scipy.stats.pearsonr, num_sys=2, by_system=True,
                            replace_nans_with_zeros=False)
    self.assertAlmostEqual(cf(v1, v2)[0], 1.0)
    cf = stats.CorrFunction(scipy.stats.pearsonr, num_sys=2, by_system=True,
                            replace_nans_with_zeros=True)
    self.assertAlmostEqual(cf(v1, v2)[0], 0.5)

  def testKendallLike(self):
    # Default
    cf = stats.KendallLike()
    res = cf.Corr([10, 11, 100, 150], [2, 1, 4, 3])
    self.assertEqual(res, (3 / 5, 5, 4, 1))

    # Thresholding.
    cf = stats.KendallLike(thresh=1)
    res = cf.Corr([10, 11, 100, 150], [2, 1, 4, 3])
    self.assertEqual(res, (2 / 6, 6, 4, 2))

    # Missing entries.
    cf = stats.KendallLike(thresh=1)
    res = cf.Corr([10, None, 100, 150], [2, 1, 4, 3])
    self.assertEqual(res, (1 / 3, 3, 2, 1))

    # Averaging.
    cf = stats.KendallLike(num_sys=2, thresh=1)
    res = cf.Corr([10, 11, 100, 150], [2, 1, 4, 3])
    self.assertEqual(res, (1, 2, 2, 0))

    # Averaging with missing entries.
    cf = stats.KendallLike(num_sys=2, thresh=1)
    res = cf.Corr([10, None, 100, 150], [2, 1, 4, 3])
    self.assertEqual(res, (1, 1, 1, 0))

  def testWilliamsSigDiff(self):
    gold = [1, 2, 3, 4, 5]
    metric1 = [1.1, 2, 3.1, 4, 5.1]
    metric2 = [1.5, 2, 3.1, 4, 5.5]
    corr1 = stats.Correlation(5, gold, metric1)
    corr2 = stats.Correlation(5, gold, metric2)

    p, r1, r2 = stats.WilliamsSigDiff(corr1, corr2, scipy.stats.pearsonr)
    self.assertAlmostEqual(p, 0.019, places=3)
    self.assertAlmostEqual(r1, 0.999, places=3)
    self.assertAlmostEqual(r2, 0.987, places=3)

    # Testing with averaging and Nones.
    gold = [1, None, 3, 4, 5, 6]
    metric1 = [1, 2, 3, 3, 7, 3]
    metric2 = [2, 1, 2, 6, 8, 8]
    corr1 = stats.Correlation(3, gold, metric1)
    corr2 = stats.Correlation(3, gold, metric2)
    corr_fcn = corr1.GenCorrFunction(scipy.stats.pearsonr, True)
    p, r1, r2 = stats.WilliamsSigDiff(corr1, corr2, corr_fcn)
    self.assertAlmostEqual(p, 0.121, places=3)
    self.assertAlmostEqual(r1, 0.982, places=3)
    self.assertAlmostEqual(r2, 0.933, places=3)


class StatsTest(unittest.TestCase):

  def testKendallPython(self):
    gold = [7, 1, 2, 7, 4, 3, 2]
    metric = [4, 1, 3, 6, 1, 5, 6]

    for variant in ['b', 'c']:
      ref = scipy.stats.kendalltau(metric, gold, variant=variant)[0]

      # Non-factored version matches scipy.
      tau = stats.KendallPython(metric, gold, variant=variant)[0]
      self.assertEqual(ref, tau)

      # Factored version matches scipy.
      prep = stats.KendallPreproc(gold)
      tau = stats.KendallPython(metric, None, preproc=prep, variant=variant)[0]
      self.assertEqual(ref, tau)

  def testReshapeAndFilter(self):
    gold = [1, 2, None, None, 5, None]
    scores1 = [1, 2, 3, 4, 5, 6]
    scores2 = [1, 2, 3, 4, 5, 6]
    corr1 = stats.Correlation(3, gold, scores1)
    corr2 = stats.Correlation(3, gold, scores2)

    lens, g, s1, s2 = stats._ReshapeAndFilter(corr1, corr2, 'sys')
    self.assertEqual(lens, [2, 1])
    self.assertEqual(list(g), [1, 2, 5])
    self.assertEqual(list(s1), [1, 2, 5])
    self.assertEqual(list(s2), [1, 2, 5])

    lens, g, s1, s2 = stats._ReshapeAndFilter(corr1, corr2, 'item')
    self.assertEqual(lens, [2, 1])
    self.assertEqual(list(g), [1, 5, 2])
    self.assertEqual(list(s1), [1, 5, 2])
    self.assertEqual(list(s2), [1, 5, 2])

    lens, g, s1, s2 = stats._ReshapeAndFilter(corr1, corr2, 'none')
    self.assertEqual(lens, [3])
    self.assertEqual(list(g), [1, 2, 5])
    self.assertEqual(list(s1), [1, 2, 5])
    self.assertEqual(list(s2), [1, 2, 5])

  def testPermutationSigDiff(self):
    gold = [1, 2, 3, 4, 5, 5, 7, 8]
    metric1 = [1, 2, 3, 3, 7, 3, 5, 5]
    metric2 = [2, 1, 2, 6, 8, 8, 7, 6]
    corr1 = stats.Correlation(4, gold, metric1)
    corr2 = stats.Correlation(4, gold, metric2)

    pearson, kendall = scipy.stats.pearsonr, scipy.stats.kendalltau

    # Pearson, no averaging, no early stopping.
    delta = pearson(metric2, gold)[0] - pearson(metric1, gold)[0]
    p, d, k = stats.PermutationSigDiff(corr1, corr2, pearson, 'none', 1000)
    self.assertGreater(p, 0)
    self.assertAlmostEqual(d, delta)
    self.assertEqual(k, 1000)

    # Pearson, no averaging, with early stopping.
    delta = pearson(metric2, gold)[0] - pearson(metric1, gold)[0]
    p, d, k = stats.PermutationSigDiff(
        corr1, corr2, pearson, 'none', 1000, block_size=50, early_max=0.2)
    self.assertGreater(p, 0)
    self.assertAlmostEqual(d, delta)
    self.assertLess(k, 1000)  # Fails with very low probabilty.

    # Pearson, system-wise averaging, no early stopping.
    cf = stats.CorrFunction(pearson, 4, by_system=True)
    delta = cf(gold, metric2)[0] - cf(gold, metric1)[0]
    p, d, k = stats.PermutationSigDiff(corr1, corr2, pearson, 'sys')
    self.assertGreater(p, 0)
    self.assertAlmostEqual(d, delta)
    self.assertEqual(k, 1000)

    # Pearson, as above but with NaN -> 0.
    cf = stats.CorrFunction(
        pearson, 4, by_system=True, replace_nans_with_zeros=True)
    delta = cf(gold, metric2)[0] - cf(gold, metric1)[0]
    p, d, k = stats.PermutationSigDiff(
        corr1, corr2, pearson, 'sys', replace_nans_with_zeros=True)
    self.assertGreater(p, 0)
    self.assertAlmostEqual(d, delta)
    self.assertEqual(k, 1000)

    # Pearson, item-wise averaging, no early stopping.
    cf = stats.CorrFunction(pearson, 4)
    delta = cf(gold, metric2)[0] - cf(gold, metric1)[0]
    p, d, k = stats.PermutationSigDiff(corr1, corr2, pearson, 'item')
    self.assertGreater(p, 0)
    self.assertAlmostEqual(d, delta)
    self.assertEqual(k, 1000)

    # Kendall, item-wise averaging, no early stopping.
    cf = stats.CorrFunction(kendall, 4)
    delta = cf(gold, metric2)[0] - cf(gold, metric1)[0]
    p, d, k = stats.PermutationSigDiff(
        corr1, corr2, kendall, 'item', k=10, block_size=10, fast_kendall=False)
    self.assertGreater(p, 0)
    self.assertAlmostEqual(d, delta)
    self.assertEqual(k, 10)
    # Factored version.
    p, d, k = stats.PermutationSigDiff(
        corr1, corr2, kendall, 'item', k=10, block_size=10, fast_kendall=True)
    self.assertGreater(p, 0)
    self.assertAlmostEqual(d, delta)
    self.assertEqual(k, 10)


if __name__ == '__main__':
  unittest.main()
