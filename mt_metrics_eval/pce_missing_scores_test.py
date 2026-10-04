# coding=utf-8
# Copyright 2026 Google LLC
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
"""Regression tests for PCE with missing segment-level gold scores."""

import functools
import unittest
from unittest import mock

from mt_metrics_eval import data
from mt_metrics_eval import pce
from mt_metrics_eval import stats
import numpy as np


class PairwiseConfidenceErrorMissingScoresTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    rng = np.random.default_rng(0)
    self.gold = rng.normal(size=(4, 40))
    self.metric = self.gold + rng.normal(size=(4, 40))
    self.other_metric = rng.normal(size=(4, 40))

  def _CheckScores(self, missing, keep):
    gold = self.gold.astype(object)
    for system, segment in missing:
      gold[system, segment] = None
    gold_scores = gold.ravel().tolist()
    metric_scores = self.metric.ravel().tolist()
    expected = pce.compute_one_minus_pce(
        pce.compute_pairwise_p_values(
            np.ascontiguousarray(self.gold[:, keep])),
        pce.compute_pairwise_p_values(
            np.ascontiguousarray(self.metric[:, keep])))

    self.assertEqual(
        stats.PairwiseConfidenceError(
            gold_scores, metric_scores, num_sys=4, filter_nones=True)[0],
        expected)
    self.assertEqual(
        stats.AverageCorrelation(stats.PairwiseConfidenceError, num_sys=4)(
            gold_scores, metric_scores)[0], expected)
    self.assertEqual(
        stats.Correlation(4, gold_scores, metric_scores)
        .PairwiseConfidenceError()[0], expected)

  def testEqualMissingCountsAtDifferentPositions(self):
    self._CheckScores(
        [(0, i) for i in range(4)] + [(1, i) for i in range(36, 40)],
        list(range(4, 36)))

  def testUnequalMissingCountsWithDivisibleTotal(self):
    self._CheckScores([(0, i) for i in range(8)], list(range(8, 40)))

  def testMissingCountNotDivisibleBySystemCount(self):
    self._CheckScores([(0, i) for i in range(3)], list(range(3, 40)))

  def testAlignedMissingColumns(self):
    self._CheckScores(
        [(system, i) for system in range(4) for i in range(5, 9)],
        list(range(5)) + list(range(9, 40)))

  def testCompleteScores(self):
    self._CheckScores([], list(range(40)))

  def testCompleteScoresPreserveFloat32ReductionOrder(self):
    rng = np.random.default_rng(1)
    gold = rng.choice([0., 1., 2., 3., 1e8, -1e8], size=(3, 10))
    metric = rng.integers(0, 5, size=(3, 10)).astype(float)
    expected = pce.compute_one_minus_pce(
        pce.compute_pairwise_p_values(gold),
        pce.compute_pairwise_p_values(metric))
    actual = stats.PairwiseConfidenceError(
        gold.ravel().tolist(), metric.ravel().tolist(), 3, filter_nones=True)
    self.assertEqual(actual[0], expected)

  def testNoCompleteSegment(self):
    gold, metric = [1, None, None, 4], [2, 3, 4, 5]
    with self.assertRaisesRegex(ValueError, 'at least one segment'):
      stats.PairwiseConfidenceError(gold, metric, 2, filter_nones=True)
    with self.assertRaisesRegex(ValueError, 'at least one segment'):
      data.CompareMetricsWithPairwiseConfidenceError(
          {'metric': stats.Correlation(2, gold, metric)}, k=0)
    with self.assertRaisesRegex(ValueError, 'at least one segment'):
      stats.PermutationSigDiff(
          stats.Correlation(2, gold, metric),
          stats.Correlation(2, gold, metric[::-1]),
          stats.PairwiseConfidenceError, num_sys=2, filter_nones=True, k=1)

  def testDirectSignificanceUsesCompleteSegments(self):
    gold = self.gold.astype(object)
    gold[0, :3] = None
    corrs, references = [], []
    for metric in (self.metric, self.other_metric):
      corrs.append(stats.Correlation(
          4, gold.ravel().tolist(), metric.ravel().tolist()))
      references.append(stats.Correlation(
          4, self.gold[:, 3:].ravel().tolist(),
          metric[:, 3:].ravel().tolist()))

    def Compare(correlations, corr_fcn, **kwargs):
      rng = np.random.RandomState(123)
      with mock.patch.object(np.random, 'binomial', side_effect=rng.binomial):
        return stats.PermutationSigDiff(
            *correlations, corr_fcn, k=16, **kwargs)

    kwargs = dict(num_sys=4, filter_nones=True)
    callbacks = [
        (stats.PairwiseConfidenceError, kwargs),
        (functools.partial(stats.PairwiseConfidenceError, **kwargs), {}),
    ]
    for corr_fcn, args in callbacks:
      with self.subTest(corr_fcn=corr_fcn):
        expected = Compare(references, corr_fcn, **args)
        self.assertEqual(Compare(corrs, corr_fcn, **args), expected)
        self.assertEqual(len(expected[3]), 16)

  def testSignificanceRejectsDifferentOriginalGoldScores(self):
    # These different masks would produce identical gold vectors if filtered
    # before validating that both correlations describe the same observations.
    first = stats.Correlation(2, [1, None, 3, 4, None, 6], list(range(6)))
    second = stats.Correlation(2, [None, 1, 3, None, 4, 6], list(range(6)))
    with self.assertRaisesRegex(ValueError, 'Gold scores'):
      stats.PermutationSigDiff(
          first, second, stats.PairwiseConfidenceError,
          num_sys=2, filter_nones=True, k=1)

  def testComparisonAndSignificanceUseCompleteSegments(self):
    cases = [
        ([(0, i) for i in range(4)] + [(1, i) for i in range(36, 40)],
         list(range(4, 36))),
        ([(0, i) for i in range(8)], list(range(8, 40))),
        ([(0, i) for i in range(3)], list(range(3, 40))),
    ]
    for missing, keep in cases:
      with self.subTest(missing=missing):
        gold = self.gold.astype(object)
        for system, segment in missing:
          gold[system, segment] = None
        actual_corrs, expected_corrs = {}, {}
        for name, metric in (
            ('metric', self.metric), ('other', self.other_metric)):
          actual_corrs[name] = stats.Correlation(
              4, gold.ravel().tolist(), metric.ravel().tolist())
          expected_corrs[name] = stats.Correlation(
              4, self.gold[:, keep].ravel().tolist(),
              metric[:, keep].ravel().tolist())

        def Compare(correlations):
          rng = np.random.RandomState(123)
          with mock.patch.object(
              np.random, 'binomial', side_effect=rng.binomial):
            return data.CompareMetricsWithPairwiseConfidenceError(
                correlations, k=16)

        expected = Compare(expected_corrs)
        actual = Compare(actual_corrs)
        self.assertEqual(actual[0], expected[0])
        for actual_array, expected_array in zip(actual[1:], expected[1:]):
          np.testing.assert_array_equal(actual_array, expected_array)
        self.assertEqual(actual[3].shape, (16, 2))
        # Filtering must not replace or alter the caller's Correlation objects.
        for name, metric in (
            ('metric', self.metric), ('other', self.other_metric)):
          self.assertEqual(
              actual_corrs[name].gold_scores, gold.ravel().tolist())
          self.assertEqual(
              actual_corrs[name].metric_scores, metric.ravel().tolist())
          self.assertEqual(actual_corrs[name].num_items, 40)
          self.assertEqual(actual_corrs[name].none_count, len(missing))


if __name__ == '__main__':
  unittest.main()
