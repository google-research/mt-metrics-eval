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
"""Tests for the tau optimization module."""

from mt_metrics_eval import tau_optimization
import numpy as np
import unittest


class TauOptimizationTest(unittest.TestCase):

  def test_zero_threshold_is_best(self):
    # There are 6 pairs:
    #    Metric: (0, 2) (0, 6) (0, 8) (2, 6) (2, 8) (6, 8)
    #    Human:  (1, 2) (1, 3) (1, 4) (2, 3) (2, 4) (3, 4)
    metric = [0, 2, 6, 8]
    human = [1, 2, 3, 4]

    expected_thresholds = [0, 2, 4, 6, 8]
    expected_taus = [6 / 6, 4 / 6, 3 / 6, 1 / 6, 0 / 6]
    expected_best_threshold = 0
    expected_best_tau = 1.0

    actual = tau_optimization.tau_optimization(
        metric, human, tau_optimization.TauSufficientStats.acc_23
    )
    self.assertEqual(expected_thresholds, actual.thresholds)
    self.assertEqual(expected_taus, actual.taus)
    self.assertEqual(expected_best_threshold, actual.best_threshold)
    self.assertEqual(expected_best_tau, actual.best_tau)

  def test_nonzero_threshold_is_best(self):
    # There are 6 pairs:
    #    Metric: (0, 2) (0, 6) (0, 8) (2, 6) (2, 8) (6, 8)
    #    Human:  (1, 1) (1, 1) (1, 4) (1, 1) (1, 4) (1, 4)
    metric = [0, 2, 6, 8]
    human = [1, 1, 1, 4]

    expected_thresholds = [0, 2, 4, 6, 8]
    expected_taus = [3 / 6, 3 / 6, 4 / 6, 4 / 6, 3 / 6]
    expected_best_threshold = 4
    expected_best_tau = 4 / 6

    actual = tau_optimization.tau_optimization(
        metric, human, tau_optimization.TauSufficientStats.acc_23
    )
    self.assertEqual(expected_thresholds, actual.thresholds)
    self.assertEqual(expected_taus, actual.taus)
    self.assertEqual(expected_best_threshold, actual.best_threshold)
    self.assertEqual(expected_best_tau, actual.best_tau)

  def test_invalid_sample_rate(self):
    metric = [0, 2, 6, 8]
    human = [1, 1, 1, 4]

    with self.assertRaises(ValueError):
      tau_optimization.tau_optimization(
          metric,
          human,
          tau_optimization.TauSufficientStats.acc_23,
          sample_rate=0.0,
      )

    with self.assertRaises(ValueError):
      tau_optimization.tau_optimization(
          metric,
          human,
          tau_optimization.TauSufficientStats.acc_23,
          sample_rate=1.1,
      )

  def test_sample_rate_samples_pairs(self):
    # Ensures that sample_rate < 1 actually downsamples pairs. The sampling
    # is random, so the random seed is fixed.
    np.random.seed(123)

    # There are (4 choose 2) = 6 pairs and each pair has a unique difference
    # in metric score. We should expect approximately half of the pairs to be
    # randomly sampled and use their diffs as thresholds (plus 0, which is
    # always considered) when sample_rate=0.5.
    metric = [0, 2, 6, 14]
    human = [1, 2, 3, 4]
    result = tau_optimization.tau_optimization(
        metric,
        human,
        tau_optimization.TauSufficientStats.acc_23,
        sample_rate=0.5,
    )
    self.assertEqual(result.thresholds, [0, 6, 8, 14])

  def test_regression_example(self):
    # Tests an example with input matrices. This result has not been manually
    # verified, but the test might catch if something changes in the
    # optimization routine.
    metric = [
        [0, 5, 2, 3, 2],
        [None, 2, 1, 5, 3],
        [9, 1, 5, 3, 8],
        [9, 3, 4, 4, 1],
        [None, None, None, None, None],
    ]
    human = [
        [1, 5, 2, 1, 1],
        [4, 2, 2, 1, 4],
        [5, 9, 2, 8, 7],
        [7, 6, None, 3, 2],
        [None, None, None, None, None],
    ]
    result = tau_optimization.tau_optimization(
        metric,
        human,
        tau_optimization.TauSufficientStats.acc_23,
    )

    expected_thresholds = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    expected_taus = [
        0.4666666666666667,
        0.4916666666666668,
        0.38333333333333347,
        0.29166666666666674,
        0.2666666666666667,
        0.2,
        0.15833333333333335,
        0.15833333333333335,
        0.1166666666666667,
    ]
    expected_best_threshold = 1
    expected_best_tau = 0.4916666666666668

    self.assertEqual(expected_thresholds, result.thresholds)
    self.assertEqual(expected_taus, result.taus)
    self.assertEqual(expected_best_threshold, result.best_threshold)
    self.assertEqual(expected_best_tau, result.best_tau)


if __name__ == "__main__":
  unittest.main()
