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
import numpy as np
import scipy.stats
import unittest

pearson = scipy.stats.pearsonr
kendall = scipy.stats.kendalltau


class CorrelationTest(unittest.TestCase):

  def testConstruction(self):
    corr = stats.Correlation(2, [1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 3, 7])
    self.assertEqual(corr.num_sys, 2)
    self.assertEqual(corr.num_items, 3)
    self.assertEqual(corr.none_count, 0)

  def testPearson(self):
    corr = stats.Correlation(2, [1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 3, 7])
    self.assertAlmostEqual(corr.Pearson(average_by='none')[0], 0.776, places=3)
    self.assertAlmostEqual(corr.Pearson(average_by='sys')[0], 0.760, places=3)
    self.assertAlmostEqual(corr.Pearson(average_by='item')[0], 1.00, places=3)

  def testKendall(self):
    corr = stats.Correlation(2, [1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 3, 7])
    self.assertAlmostEqual(corr.Kendall(average_by='none')[0], 0.552, places=3)
    self.assertAlmostEqual(corr.Kendall(average_by='sys')[0], 0.575, places=3)
    self.assertAlmostEqual(corr.Kendall(average_by='item')[0], 1.00, places=3)

  def testKendallC(self):
    corr = stats.Correlation(2, [1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 3, 7])
    self.assertAlmostEqual(
        corr.Kendall(average_by='none', variant='c')[0], 0.556, places=3)
    self.assertAlmostEqual(
        corr.Kendall(average_by='sys', variant='c')[0], 0.611, places=3)
    self.assertAlmostEqual(
        corr.Kendall(average_by='item', variant='c')[0], 1.00, places=3)

  def testKendallLike(self):
    corr = stats.Correlation(1, [10, None, 100, 150], [2, 1, 4, 3])
    res = corr.KendallLike(thresh=1, average_by='none')
    self.assertEqual(res[0], 1 / 3)

    corr = stats.Correlation(2, [10, 11, 100, 150], [2, 1, 4, 3])
    res = corr.KendallLike(average_by='item', thresh=1)
    self.assertEqual(res[0], 1)

    corr = stats.Correlation(2, [10, None, 100, 150], [2, 1, 4, 3])
    res = corr.KendallLike(average_by='item', thresh=1)
    self.assertEqual(res[0], 1)

  def testKendallVariants(self):
    corr = stats.Correlation(1, [1, 2, 2, 3, 4], [2, 1, 1.5, 5, 3])
    self.assertAlmostEqual(corr.KendallVariants(variant='b')[0], 0.316227766)
    self.assertEqual(corr.KendallVariants(variant='c')[0], 0.32)
    self.assertEqual(corr.KendallVariants(variant='23')[0], 0.2)
    self.assertEqual(corr.KendallVariants(variant='acc23')[0], 0.6)
    self.assertEqual(corr.KendallVariants(variant='acc23', epsilon=2)[0], 0.4)

  def testKendallWithTiesOpt(self):
    corr = stats.Correlation(1, [1, 2, 2, 3, 4], [2, 1, 1.5, 5, 3])
    self.assertEqual(
        corr.KendallWithTiesOpt(variant='acc23', sample_rate=1.0)[0], 0.7)


class AverageCorrelationTest(unittest.TestCase):

  def testAverageByNone(self):
    g, m = [1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 5, 6]
    cf = stats.AverageCorrelation(pearson)
    self.assertEqual(cf(g, m)[:2], pearson(g, m))

  def testAverageByNoneNoFiltering(self):
    g, m = [1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 5, 6]
    cf = stats.AverageCorrelation(pearson, filter_nones=False)
    self.assertEqual(cf(g, m)[:2], pearson(g, m))

  def testMissingEntries(self):
    g, m = [1, None, 3, 4, 5, 6], [2, 1, 4, 3, 5, 6]
    ge, me = [1, 3, 4, 5, 6], [2, 4, 3, 5, 6]
    cf = stats.AverageCorrelation(pearson)
    self.assertEqual(cf(g, m)[:2], pearson(ge, me))

  def testAverageBySystem(self):
    g, m = [1, 2, 3, 4, 5, 6], [2, 1, 8, 3, 5, 6]
    cf = stats.AverageCorrelation(
        pearson, num_sys=2, average_by='sys')
    p1 = pearson([1, 2, 3], [2, 1, 8])
    p2 = pearson([4, 5, 6], [3, 5, 6])
    c12 = cf(g, m)
    self.assertAlmostEqual(c12[0], (p1[0] + p2[0]) / 2, places=3)
    self.assertAlmostEqual(c12[1], (p1[1] + p2[1]) / 2, places=3)

  def testAverageByItem(self):
    g, m = [1, 2, 3, 4, 5, 6], [2, 1, 8, 3, 5, 6]
    cf = stats.AverageCorrelation(
        pearson, num_sys=2, average_by='item')
    p1 = pearson([1, 4], [2, 3])
    p2 = pearson([2, 5], [1, 5])
    p3 = pearson([3, 6], [8, 6])
    c12 = cf(g, m)
    self.assertAlmostEqual(c12[0], (p1[0] + p2[0] + p3[0]) / 3, places=3)
    self.assertAlmostEqual(c12[1], (p1[1] + p2[1] + p3[1]) / 3, places=3)

  def testAverageByItemWithMissingEntries(self):
    g, m = [1, None, 3, 4, 5, 6], [2, 1, 8, 3, 5, 6]
    cf = stats.AverageCorrelation(
        pearson, num_sys=2, average_by='item')
    p1 = pearson([1, 4], [2, 3])
    p3 = pearson([3, 6], [8, 6])
    c12 = cf(g, m)
    self.assertAlmostEqual(c12[0], (p1[0] + p3[0]) / 2, places=3)
    self.assertAlmostEqual(c12[1], (p1[1] + p3[1]) / 2, places=3)

  def testNaNHandling(self):
    g, m = [1, 2, 3, 4, 5, 6], [2, 2, 2, 4, 5, 6]
    cf = stats.AverageCorrelation(
        pearson, num_sys=2, average_by='sys',
        replace_nans_with_zeros=False)
    self.assertAlmostEqual(cf(g, m)[0], 1.0)

  def testNaNHandlingWithZeros(self):
    g, m = [1, 2, 3, 4, 5, 6], [2, 2, 2, 4, 5, 6]
    cf = stats.AverageCorrelation(
        pearson, num_sys=2, average_by='sys',
        replace_nans_with_zeros=True)
    self.assertAlmostEqual(cf(g, m)[0], 0.5)

  def testMicroAveraging(self):
    g, m = [1, None, 3, 4, 5, 6], [2, 1, 8, 3, 5, 6]
    cf = stats.AverageCorrelation(
        pearson, num_sys=2, average_by='sys', macro=False)
    p1 = pearson([1, 3], [2, 8])
    p2 = pearson([4, 5, 6], [3, 5, 6])
    c12 = cf(g, m)
    self.assertAlmostEqual(c12[0], (2 * p1[0] + 3 * p2[0]) / 5, places=3)
    self.assertAlmostEqual(c12[1], (2 * p1[1] + 3 * p2[1]) / 5, places=3)

  def testCorrFcnArgs(self):
    g, m = [1, 2, 3, 4, 5, 6], [2, 1, 8, 3, 5, 6]
    cf = stats.AverageCorrelation(kendall, average_by='none', variant='b')
    self.assertEqual(cf(g, m)[:2], kendall(g, m, variant='b'))
    cf = stats.AverageCorrelation(kendall, average_by='none', variant='c')
    self.assertEqual(cf(g, m)[:2], kendall(g, m, variant='c'))

  def testKendallWithTiesOpt(self):
    g, m = [1, 2, 3, 4, None, 6], [2, 1, 8, 3, 5, 6]
    res = stats.AverageCorrelation(
        stats.KendallWithTiesOpt, num_sys=2, average_by='sys', variant='acc23',
        sample_rate=1.0)(g, m)
    ref = stats.KendallWithTiesOpt(
        g, m, num_sys=2, average_by='sys', variant='acc23', sample_rate=1.0)
    self.assertEqual(res[:2], ref[:2])


class CorrelationFunctionsTest(unittest.TestCase):

  def testKendallLike(self):
    gold, metric = [10, 11, 100, 150], [2, 1, 4, 3]
    self.assertEqual(stats.KendallLike(gold, metric), (3 / 5, 0, 5, 4, 1))
    self.assertEqual(stats.KendallLike(gold, metric, 1), (2 / 6, 0, 6, 4, 2))

  def testAgreement(self):
    gold, metric = [1, 2, 3, 3, 4], [1, 1, 2, 1, 2]
    agree, num_pairs = stats.Agreement(gold, metric)
    self.assertEqual(agree, 5)
    self.assertEqual(num_pairs, 10)

  def testKendallVariants_b_and_c(self):
    gold = [7, 1, 2, 7, 4, 3, 2]
    metric = [4, 1, 3, 6, 1, 5, 6]

    for variant in ['b', 'c']:
      ref = kendall(gold, metric, variant=variant)[0]

      # Non-factored version matches scipy.
      tau = stats.KendallVariants(gold, metric, variant=variant)[0]
      self.assertEqual(ref, tau)

      # Factored version matches scipy.
      prep = stats.KendallPreproc(gold)
      tau = stats.KendallVariants(
          None, metric, preproc=prep, variant=variant)[0]
      self.assertEqual(ref, tau)

      # Metric-factored version matches scipy
      pd = stats.PairwiseDiffs(metric)
      m = metric if variant == 'c' else None
      tau = stats.KendallVariants(
          gold, m, variant=variant, metric_preproc=pd)[0]
      self.assertEqual(ref, tau)

      # Dual-factored version matches scipy
      m = metric if variant == 'c' else None
      tau = stats.KendallVariants(
          None, m, variant=variant, preproc=prep, metric_preproc=pd)[0]
      self.assertEqual(ref, tau)

  def testKendallVariants_23(self):
    x = [1, 1, 1, 2, 2, 3, 4]
    y = [1, 1, 2, 2, 4, 3, 3]

    num_pairs = 21  # (7 choose 2)
    ties_in_x = 4  # (1, 1), (1, 1), (1, 1), (2, 2)
    ties_in_y = 3  # (1, 1), (2, 2), (3, 3)
    ties_in_both = 1  # the first (1, 1) in each
    ties_in_x_only = ties_in_x - ties_in_both
    ties_in_y_only = ties_in_y - ties_in_both
    discordant = 2  # (2, 3) versus (4, 3) and (2, 4) versus (4, 3)
    # All others are concordant
    concordant = (
        num_pairs - discordant - ties_in_x_only - ties_in_y_only - ties_in_both
    )

    expected_tau = (
        concordant + ties_in_both - discordant - ties_in_x_only - ties_in_y_only
    ) / num_pairs
    actual_tau = stats.KendallVariants(x, y, variant='23')[0]
    self.assertEqual(actual_tau, expected_tau)

    expected_acc = (concordant + ties_in_both) / num_pairs
    actual_acc = stats.KendallVariants(x, y, variant='acc23')[0]
    self.assertEqual(actual_acc, expected_acc)

  def testKendallVariantsWithPositiveEpsilon(self):
    x = [1, 1, 1, 2, 2, 3, 4]
    y = [1, 1, 2, 2, 4, 3, 3]

    # Epsilon is too small to introduce any ties, so this should be the
    # same as when epsilon = 0
    expected_tau = (14 - 7) / 21  # see `testKendallVariants_23`
    actual_tau = stats.KendallVariants(x, y, variant='23', epsilon=0.0)[0]
    self.assertEqual(actual_tau, expected_tau)
    actual_tau = stats.KendallVariants(x, y, variant='23', epsilon=0.1)[0]
    self.assertEqual(actual_tau, expected_tau)

    # Introduce ties in x with epsilon = 1
    num_pairs = 21
    ties_in_x = 13  # (1, 1) x 3, (1, 2) x 6, (2, 2), (2, 3) x 2, and (3, 4)
    ties_in_y = 3  # (1, 1), (2, 2), (3, 3)
    ties_in_both = 3  # All of y's ties are now tied in x
    ties_in_x_only = ties_in_x - ties_in_both
    ties_in_y_only = ties_in_y - ties_in_both
    discordant = 1  # (2, 4) versus (4, 3)
    # All others are concordant
    concordant = (
        num_pairs - discordant - ties_in_x_only - ties_in_y_only - ties_in_both
    )
    expected_tau = (
        concordant + ties_in_both - discordant - ties_in_x_only - ties_in_y_only
    ) / num_pairs
    actual_tau = stats.KendallVariants(x, y, variant='23', epsilon=1.0)[0]
    self.assertEqual(actual_tau, expected_tau)

    # Test factored versions
    prep = stats.KendallPreproc(x)
    pd = stats.PairwiseDiffs(y, epsilon=1.0)
    #
    actual_tau = stats.KendallVariants(
        None, y, epsilon=1.0, preproc=prep, variant='23')[0]
    self.assertEqual(actual_tau, expected_tau)
    #
    actual_tau = stats.KendallVariants(
        x, None, metric_preproc=pd, variant='23')[0]
    self.assertEqual(actual_tau, expected_tau)
    #
    actual_tau = stats.KendallVariants(
        None, None, preproc=prep, metric_preproc=pd, variant='23')[0]
    self.assertEqual(actual_tau, expected_tau)

  def testKendallWithTiesOpt(self):
    gold, metric = [1, 2, 2, 3, 3, 4], [2, 1, 1.5, 5, 3, 2]

    corr, eps, _ = stats.KendallWithTiesOpt(
        gold, metric, variant='acc23', num_sys=1, average_by='none',
        sample_rate=1.0)
    ref = stats.KendallVariants(
        gold, metric, variant='acc23', epsilon=eps)[0]
    self.assertEqual(corr, ref)

    corr, eps, _ = stats.KendallWithTiesOpt(
        gold, metric, variant='acc23', num_sys=2, average_by='sys',
        sample_rate=1.0)
    gold1, gold2 = gold[:3], gold[3:]
    metric1, metric2 = metric[:3], metric[3:]
    r1 = stats.KendallVariants(gold1, metric1, variant='acc23', epsilon=eps)[0]
    r2 = stats.KendallVariants(gold2, metric2, variant='acc23', epsilon=eps)[0]
    self.assertEqual(corr, (r1 + r2) / 2)

    corr, eps, _ = stats.KendallWithTiesOpt(
        gold, metric, variant='acc23', num_sys=2, average_by='item',
        sample_rate=1.0)
    gold = np.array(gold).reshape((2, 3)).transpose()
    metric = np.array(metric).reshape((2, 3)).transpose()
    gold1, gold2, gold3 = gold[0], gold[1], gold[2]
    metric1, metric2, metric3 = metric[0], metric[1], metric[2]
    r1 = stats.KendallVariants(gold1, metric1, variant='acc23', epsilon=eps)[0]
    r2 = stats.KendallVariants(gold2, metric2, variant='acc23', epsilon=eps)[0]
    r3 = stats.KendallVariants(gold3, metric3, variant='acc23', epsilon=eps)[0]
    self.assertEqual(corr, (r1 + r2 + r3) / 3)

  def testReshape(self):
    vect = [1, 2, 3, 4, 5, 6]
    self.assertEqual(stats._Reshape(vect, 2, 'none').shape, (1, 6))
    self.assertEqual(stats._Reshape(vect, 2, 'sys').shape, (2, 3))
    self.assertEqual(stats._Reshape(vect, 2, 'item').shape, (3, 2))

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


class PermutationSigDiffTest(unittest.TestCase):

  gold = [1, 2, 3, 4, 5, 5, 7, 8]
  metric1 = [1, 2, 3, 3, 7, 3, 5, 5]
  metric2 = [2, 1, 2, 6, 8, 8, 7, 6]
  corr1 = stats.Correlation(4, gold, metric1)
  corr2 = stats.Correlation(4, gold, metric2)

  def testPearsonNoAvgNoEarlyStop(self):
    delta = (pearson(self.metric2, self.gold)[0] -
             pearson(self.metric1, self.gold)[0])
    p, d, k, c = stats.PermutationSigDiff(
        self.corr1, self.corr2, pearson, 'none', 1000)
    self.assertGreater(p, 0)
    self.assertAlmostEqual(d, delta)
    self.assertEqual(k, 1000)
    self.assertEqual(len(c), 1000)  # pylint: disable=g-generic-assert
    self.assertAlmostEqual(sum(c2 - c1 >= delta for c2, c1 in c) / 1000, p)

  def testPearsonNoAvgWithEarlyStop(self):
    delta = (pearson(self.metric2, self.gold)[0] -
             pearson(self.metric1, self.gold)[0])
    p, d, k, c = stats.PermutationSigDiff(
        self.corr1, self.corr2, pearson, 'none', 1000,
        stats.PermutationSigDiffParams(block_size=50, early_max=0.2))
    self.assertGreater(p, 0)
    self.assertAlmostEqual(d, delta)
    self.assertLess(k, 1000)  # Fails with very low probabilty.
    self.assertEqual(len(c), k)  # pylint: disable=g-generic-assert
    self.assertAlmostEqual(sum(c2 - c1 >= delta for c2, c1 in c) / k, p)

  def testPearsonSysNoEarlyStop(self):
    cf = stats.AverageCorrelation(pearson, 4, average_by='sys')
    delta = cf(self.gold, self.metric2)[0] - cf(self.gold, self.metric1)[0]
    p, d, k, _ = stats.PermutationSigDiff(
        self.corr1, self.corr2, pearson, 'sys')
    self.assertGreater(p, 0)
    self.assertAlmostEqual(d, delta)
    self.assertEqual(k, 1000)

  def testPearsonSysNoEarlyStopNan0(self):
    cf = stats.AverageCorrelation(
        pearson, 4, average_by='sys', replace_nans_with_zeros=True)
    delta = cf(self.gold, self.metric2)[0] - cf(self.gold, self.metric1)[0]
    p, d, k, _ = stats.PermutationSigDiff(
        self.corr1, self.corr2, pearson, 'sys', replace_nans_with_zeros=True)
    self.assertGreater(p, 0)
    self.assertAlmostEqual(d, delta)
    self.assertEqual(k, 1000)

  def testPearsonItemNoEarlyStop(self):
    cf = stats.AverageCorrelation(pearson, 4, average_by='item')
    delta = cf(self.gold, self.metric2)[0] - cf(self.gold, self.metric1)[0]
    p, d, k, _ = stats.PermutationSigDiff(
        self.corr1, self.corr2, pearson, 'item')
    self.assertGreater(p, 0)
    self.assertAlmostEqual(d, delta)
    self.assertEqual(k, 1000)

  def testKendallItemWithEarlyStop(self):
    cf = stats.AverageCorrelation(kendall, 4, average_by='item')
    delta = cf(self.gold, self.metric2)[0] - cf(self.gold, self.metric1)[0]
    p, d, k, _ = stats.PermutationSigDiff(
        self.corr1, self.corr2, kendall, 'item', k=10,
        params=stats.PermutationSigDiffParams(block_size=10))
    self.assertGreater(p, 0)
    self.assertAlmostEqual(d, delta)
    self.assertEqual(k, 10)
    # Factored version.
    p, d, k, _ = stats.PermutationSigDiff(
        self.corr1, self.corr2, stats.KendallVariants, 'item', k=10,
        params=stats.PermutationSigDiffParams(block_size=10))
    self.assertGreater(p, 0)
    self.assertAlmostEqual(d, delta)
    self.assertEqual(k, 10)

  def testKendallC(self):
    tau1 = self.corr1.Kendall(average_by='item', variant='c')[0]
    tau2 = self.corr2.Kendall(average_by='item', variant='c')[0]
    p, d, k, _ = stats.PermutationSigDiff(
        self.corr1, self.corr2, kendall, 'item', k=10,
        params=stats.PermutationSigDiffParams(block_size=10), variant='c')
    self.assertGreater(p, 0)
    self.assertAlmostEqual(d, tau2 - tau1)
    self.assertEqual(k, 10)

  def testKendallVariants23(self):
    tau1 = self.corr1.KendallVariants(average_by='item', variant='23')[0]
    tau2 = self.corr2.KendallVariants(average_by='item', variant='23')[0]
    p, d, k, _ = stats.PermutationSigDiff(
        self.corr1, self.corr2, stats.KendallVariants, 'item', k=10,
        params=stats.PermutationSigDiffParams(block_size=10), variant='23')
    self.assertGreater(p, 0)
    self.assertAlmostEqual(d, tau2 - tau1)
    self.assertEqual(k, 10)

  def testKendallVariantsAcc23(self):
    tau1 = self.corr1.KendallVariants(average_by='item', variant='acc23')[0]
    tau2 = self.corr2.KendallVariants(average_by='item', variant='acc23')[0]
    p, d, k, _ = stats.PermutationSigDiff(
        self.corr1, self.corr2, stats.KendallVariants, 'item', k=10,
        params=stats.PermutationSigDiffParams(block_size=10), variant='acc23')
    self.assertGreater(p, 0)
    self.assertAlmostEqual(d, tau2 - tau1)
    self.assertEqual(k, 10)

  def testTauOptimization(self):
    tau1 = self.corr1.KendallWithTiesOpt(
        average_by='item', variant='acc23', sample_rate=1.0)[0]
    tau2 = self.corr2.KendallWithTiesOpt(
        average_by='item', variant='acc23', sample_rate=1.0)[0]
    p, d, k, _ = stats.PermutationSigDiff(
        self.corr1, self.corr2, stats.KendallWithTiesOpt,
        average_by='item', k=10,
        params=stats.PermutationSigDiffParams(block_size=10),
        variant='acc23', sample_rate=1.0)
    self.assertGreater(p, 0)
    self.assertAlmostEqual(d, tau2 - tau1)
    self.assertEqual(k, 10)


class PairwisePermutationSigDiffTest(unittest.TestCase):

  gold = [1, 3, 3, 3, 1, 2, 1, 2]
  metric1 = [2, 4, 6, 6, 2, 6, 3, 7]
  metric2 = [1, 5, 4, 4, 4, 3, 2, 1]
  corr1 = stats.Correlation(4, gold, metric1)
  corr2 = stats.Correlation(4, gold, metric2)

  def testKendallVariantsAcc23(self):
    tau1 = self.corr1.KendallWithTiesOpt('item', 'acc23', sample_rate=1.0)[0]
    tau2 = self.corr2.KendallWithTiesOpt('item', 'acc23', sample_rate=1.0)[0]
    delta = tau2 - tau1
    p, d, k, c = stats.PairwisePermutationSigDiff(
        self.corr1, self.corr2, 'acc23', 'item', k=10, sample_rate=1.0)
    self.assertGreater(p, 0)
    self.assertAlmostEqual(d, delta)
    self.assertEqual(k, 10)
    self.assertEqual(len(c), 10)  # pylint: disable=g-generic-assert

    # Test can flake due to very small differences relative to delta, so use
    # a tolerance of +- 3 when comparing null counts.
    expected_null_count = p * k
    null_count = sum(c2 - c1 >= delta for c2, c1 in c)
    self.assertLessEqual(abs(null_count - expected_null_count), 3)

  def testKendallVariantsB(self):
    tau1 = self.corr1.KendallVariants('item', 'b', epsilon=1)[0]
    tau2 = self.corr2.KendallVariants('item', 'b', epsilon=1)[0]
    _, d, k, _ = stats.PairwisePermutationSigDiff(
        self.corr1, self.corr2, 'b', 'item', k=10, epsilon1=1, epsilon2=1)
    self.assertAlmostEqual(d, tau2 - tau1)
    self.assertEqual(k, 10)


class WilliamsSigDiffTest(unittest.TestCase):

  def testSigDiff(self):
    gold = [1, 2, 3, 4, 5]
    metric1 = [1.1, 2, 3.1, 4, 5.1]
    metric2 = [1.5, 2, 3.1, 4, 5.5]
    corr1 = stats.Correlation(5, gold, metric1)
    corr2 = stats.Correlation(5, gold, metric2)

    p, r1, r2 = stats.WilliamsSigDiff(corr1, corr2, pearson)
    self.assertAlmostEqual(p, 0.019, places=3)
    self.assertAlmostEqual(r1, 0.999, places=3)
    self.assertAlmostEqual(r2, 0.987, places=3)

  def testSigDiffWithAvgAndNones(self):
    gold = [1, None, 3, 4, 5, 6]
    metric1 = [1, 2, 3, 3, 7, 3]
    metric2 = [2, 1, 2, 6, 8, 8]
    corr1 = stats.Correlation(3, gold, metric1)
    corr2 = stats.Correlation(3, gold, metric2)
    corr_fcn = stats.AverageCorrelation(pearson, 3, average_by='item')
    p, r1, r2 = stats.WilliamsSigDiff(corr1, corr2, corr_fcn)
    self.assertAlmostEqual(p, 0.121, places=3)
    self.assertAlmostEqual(r1, 0.982, places=3)
    self.assertAlmostEqual(r2, 0.933, places=3)


class SampleTest(unittest.TestCase):

  def testUniformSample(self):
    s1 = stats.Sample(10, 5, 'uniform', seed=11)
    self.assertEqual(len(s1.sample), 5)  # pylint: disable=g-generic-assert
    s2 = stats.Sample(10, 5, 'uniform', seed=11)
    self.assertEqual(list(s1.sample), list(s2.sample))

    self.assertEqual(list(s1.sample), list(s1.Select(range(10))))

  def testStratifiedSample(self):
    s = stats.Sample(8, 4, 'stratify', [2, 2, 2, 2])
    self.assertEqual(len(s.sample), 4)  # pylint: disable=g-generic-assert
    self.assertIn(s.sample[0], [0, 1])
    self.assertIn(s.sample[1], [2, 3])
    self.assertIn(s.sample[2], [4, 5])
    self.assertIn(s.sample[3], [6, 7])

    s = stats.Sample(8, 4, 'stratify', [2, 6])
    self.assertEqual(len(s.sample), 4)  # pylint: disable=g-generic-assert
    self.assertIn(s.sample[0], [0, 1])
    self.assertIn(s.sample[1], range(2, 8))
    self.assertIn(s.sample[2], range(2, 8))
    self.assertIn(s.sample[3], range(2, 8))

    s = stats.Sample(8, 3, 'stratify', [1, 7])
    self.assertEqual(len(s.sample), 3)  # pylint: disable=g-generic-assert
    self.assertNotIn(0, s.sample)


class PairwiseDiffsTest(unittest.TestCase):
  def testPairWiseDiffs(self):
    x1 = stats.PairwiseDiffs([1.0, 2.0, 2.1, 3.0], 0.2)
    x2 = stats.PairwiseDiffs([4.0, 2.0, 3.0, 5.0], 1.0)

    # Number of actual ties, correcting for diagonal and double counting.
    self.assertEqual((x1.x_is_tie.sum() - 4) / 2, 1)
    self.assertEqual((x2.x_is_tie.sum() - 4) / 2, 3)

    # Number of > and < comparisons.
    self.assertEqual((np.triu(x1.x_diffs) > 0).sum(), 5)
    self.assertEqual((np.triu(x1.x_diffs) < 0).sum(), 0)
    self.assertEqual((np.triu(x2.x_diffs) > 0).sum(), 2)
    self.assertEqual((np.triu(x2.x_diffs) < 0).sum(), 1)

    # Combine with complementary weight matrices
    w1 = np.triu(np.random.binomial(1, 0.5, (4, 4)))
    w1 = w1 + w1.T
    w2 = 1 - w1
    x1h = x1.Combine(x2, w1, w2)
    x2h = x2.Combine(x1, w1, w2)

    # Ties are conserved
    nties1 = (x1h.x_is_tie.sum() - 4) / 2
    nties2 = (x2h.x_is_tie.sum() - 4) / 2
    self.assertEqual(nties1 + nties2, 4)

    # > and < are conserved
    ngt = (np.triu(x1h.x_diffs) > 0).sum() + (np.triu(x2h.x_diffs) > 0).sum()
    nlt = (np.triu(x1h.x_diffs) < 0).sum() + (np.triu(x2h.x_diffs) < 0).sum()
    self.assertEqual(ngt, 7)
    self.assertEqual(nlt, 1)


if __name__ == '__main__':
  unittest.main()
