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
"""Seeded sampling must be independent of NumPy's global random state."""

from absl.testing import parameterized
from mt_metrics_eval import stats
import numpy as np
import unittest


class SeededSampleTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    state = np.random.get_state()
    self.addCleanup(np.random.set_state, state)

  @parameterized.parameters(0, 7, 11)
  def test_stratified_sample_ignores_global_seed(self, seed):
    samples = []
    for global_seed in range(6):
      np.random.seed(global_seed)
      samples.append(stats.Sample(12, 5, 'stratify', [4, 4, 4], seed).sample)
    for sample in samples[1:]:
      self.assertEqual(list(sample), list(samples[0]))

  @parameterized.parameters(0, 7, 11)
  def test_repeated_stratified_draws_use_only_explicit_seed(self, seed):
    np.random.seed(1)
    expected = stats.Sample(12, 5, 'stratify', [4, 4, 4], seed).sample
    for _ in range(5):
      np.random.random(23)
      actual = stats.Sample(12, 5, 'stratify', [4, 4, 4], seed).sample
      self.assertEqual(list(actual), list(expected))

  @parameterized.parameters(0, 11, None)
  def test_stratified_sample_does_not_advance_global_rng(self, seed):
    np.random.seed(876)
    expected = np.random.random(10)
    np.random.seed(876)
    stats.Sample(12, 5, 'stratify', [4, 4, 4], seed)
    np.testing.assert_array_equal(np.random.random(10), expected)

  @parameterized.parameters(0, 11)
  def test_uniform_sample_matches_seeded_generator(self, seed):
    expected = np.random.default_rng(seed).choice(100, 12, replace=False)
    sample = stats.Sample(100, 12, 'uniform', seed=seed)
    np.testing.assert_array_equal(sample.sample, expected)
    np.testing.assert_array_equal(
        sample.Select(np.arange(100) * 3), expected * 3)

  @parameterized.parameters(0, 11)
  def test_stratified_allocation_and_selection_share_generator(self, seed):
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(3)
    sizes = np.ones(3, dtype=int)
    residuals = np.full(3, -2 / 3)
    sizes[np.argpartition(residuals, 2)[:2]] += 1
    sizes = sizes[np.argsort(shuffled)]
    expected = []
    for index, size in enumerate(sizes):
      expected.extend(rng.choice(4, size, replace=False) + index * 4)
    np.random.seed(1)
    actual = stats.Sample(12, 5, 'stratify', [4, 4, 4], seed)
    self.assertEqual(list(actual.sample), expected)

  @parameterized.parameters(
      ([2, 2, 2, 2], 4), ([1, 7], 3), ([0, 4, 8], 5),
      ([4, 4, 4], 0), ([4, 4, 4], 20))
  def test_stratified_sample_preserves_allocation_constraints(self, bins, size):
    total = sum(bins)
    sample = stats.Sample(total, size, 'stratify', bins, seed=0)
    expected_size = total if size == 0 or size > total else size
    self.assertLen(sample.sample, expected_size)
    self.assertLen(set(sample.sample), expected_size)
    self.assertTrue(all(0 <= index < total for index in sample.sample))
    counts = np.histogram(sample.sample, np.cumsum([0] + bins))[0]
    ideal = np.asarray(bins) * expected_size / total
    self.assertTrue(np.all(np.abs(counts - ideal) < 1))
    self.assertTrue(np.all(counts <= bins))

  def test_distinct_stratified_seeds_are_not_collapsed(self):
    samples = {
        tuple(stats.Sample(60, 20, 'stratify', [20, 20, 20], seed).sample)
        for seed in range(5)
    }
    self.assertLen(samples, 5)


if __name__ == '__main__':
  unittest.main()
