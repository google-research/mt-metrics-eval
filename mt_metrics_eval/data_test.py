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
"""Tests for data module."""

from mt_metrics_eval import data
from mt_metrics_eval import meta_info
import numpy as np
import unittest


class EvalSetTest(unittest.TestCase):

  def _std_sys_names(self, evs):
    return evs.sys_names - evs.human_sys_names - evs.outlier_sys_names

  def testWMT20EnDeSysCorrelations(self):
    evs = data.EvalSet('wmt20', 'en-de', True)
    # Spot-checking table 6 in www.statmt.org/wmt20/pdf/2020.wmt-1.77.pdf
    results = {}
    sys_names = self._std_sys_names(evs)
    gold_scores = evs.Scores('sys', evs.StdHumanScoreName('sys'))
    for m in 'BLEU', 'sentBLEU', 'COMET', 'BLEURT-extended', 'prism', 'YiSi-0':
      metric_scores = evs.Scores('sys', m + '-ref')
      results[m] = evs.Correlation(
          gold_scores, metric_scores, sys_names).Pearson()[0]
    self.assertAlmostEqual(results['BLEU'], 0.825, places=3)
    self.assertAlmostEqual(results['sentBLEU'], 0.823, places=3)
    self.assertAlmostEqual(results['COMET'], 0.863, places=3)
    self.assertAlmostEqual(results['BLEURT-extended'], 0.870, places=3)
    self.assertAlmostEqual(results['prism'], 0.851, places=3)
    self.assertAlmostEqual(results['YiSi-0'], 0.889, places=3)

    # Spot-checking table 7 in www.statmt.org/wmt20/pdf/2020.wmt-1.77.pdf, 3rd
    # column.
    sys_names.add('ref')
    for m in 'BLEU', 'sentBLEU', 'COMET', 'BLEURT-extended', 'prism', 'YiSi-0':
      variant_name = m + '-refb'
      metric_scores = evs.Scores('sys', variant_name)
      results[variant_name] = evs.Correlation(
          gold_scores, metric_scores, sys_names).Pearson()[0]
    self.assertAlmostEqual(results['BLEU-refb'], 0.672, places=3)
    self.assertAlmostEqual(results['sentBLEU-refb'], 0.639, places=3)
    self.assertAlmostEqual(results['COMET-refb'], 0.879, places=3)
    self.assertAlmostEqual(results['BLEURT-extended-refb'], 0.883, places=3)
    self.assertAlmostEqual(results['prism-refb'], 0.731, places=3)
    self.assertAlmostEqual(results['YiSi-0-refb'], 0.728, places=3)

  def testWMT20EnDeDocCorrelations(self):
    evs = data.EvalSet('wmt20', 'en-de', True)
    # Spot-checking table 12 in www.statmt.org/wmt20/pdf/2020.wmt-1.77.pdf, 4th
    # column (numbers do not match the table, ones here are correct).
    results = {}
    sys_names = self._std_sys_names(evs)
    gold_scores = evs.Scores('doc', evs.StdHumanScoreName('doc'))
    for m in 'sentBLEU', 'COMET', 'BLEURT-extended', 'prism', 'YiSi-0':
      metric_scores = evs.Scores('doc', m + '-ref')
      corr = evs.Correlation(gold_scores, metric_scores, sys_names)
      c, _, num_pairs = corr.KendallLike()
      self.assertEqual(num_pairs, 275)
      results[m] = c
    self.assertAlmostEqual(results['sentBLEU'], 0.411, places=3)
    self.assertAlmostEqual(results['COMET'], 0.433, places=3)
    self.assertAlmostEqual(results['BLEURT-extended'], 0.396, places=3)
    self.assertAlmostEqual(results['prism'], 0.389, places=3)
    self.assertAlmostEqual(results['YiSi-0'], 0.360, places=3)

  def testWMT20EnDeSegCorrelations(self):
    evs = data.EvalSet('wmt20', 'en-de', True)
    # Spot-checking table 10 in www.statmt.org/wmt20/pdf/2020.wmt-1.77.pdf, 4th
    # column.
    results = {}
    sys_names = self._std_sys_names(evs)
    gold_scores = evs.Scores('seg', evs.StdHumanScoreName('seg'))
    for m in 'sentBLEU', 'COMET', 'BLEURT-extended', 'prism', 'YiSi-0':
      metric_scores = evs.Scores('seg', m + '-ref')
      corr = evs.Correlation(gold_scores, metric_scores, sys_names)
      c, _, num_pairs = corr.KendallLike()
      results[m] = c
      self.assertEqual(num_pairs, 4637)
    self.assertAlmostEqual(results['sentBLEU'], 0.155, places=3)
    self.assertAlmostEqual(results['COMET'], 0.324, places=3)
    self.assertAlmostEqual(results['BLEURT-extended'], 0.278, places=3)
    self.assertAlmostEqual(results['prism'], 0.280, places=3)
    self.assertAlmostEqual(results['YiSi-0'], 0.212, places=3)

  def testWMT20EnDeMQMScores(self):
    evs = data.EvalSet('wmt20', 'en-de')
    results = {}
    for level in 'sys', 'doc', 'seg':
      scores = evs.Scores(level, 'mqm')
      n = len(scores)
      self.assertEqual(n, 10)
      results[level] = (scores['OPPO.1535'][0], scores['OPPO.1535'][-1])
    self.assertAlmostEqual(results['sys'][0], -2.24805, places=5)
    self.assertAlmostEqual(results['sys'][1], -2.24805, places=5)
    self.assertAlmostEqual(results['doc'][0], -1.55128, places=5)
    self.assertAlmostEqual(results['doc'][1], -1.26429, places=5)
    self.assertAlmostEqual(results['seg'][0], -2.66667, places=5)
    self.assertAlmostEqual(results['seg'][1], -1.33333, places=5)

  def testWMT20EnDePSQMScores(self):
    evs = data.EvalSet('wmt20', 'en-de')
    results = {}
    for level in 'sys', 'doc', 'seg':
      scores = evs.Scores(level, 'psqm')
      n = len(scores)
      self.assertEqual(n, 10)
      results[level] = (scores['OPPO.1535'][0], scores['OPPO.1535'][-1])
    self.assertAlmostEqual(results['sys'][0], 3.78561, places=5)
    self.assertAlmostEqual(results['sys'][1], 3.78561, places=5)
    self.assertAlmostEqual(results['doc'][0], 4.41026, places=5)
    self.assertAlmostEqual(results['doc'][1], 4.38095, places=5)
    self.assertAlmostEqual(results['seg'][0], 4.0, places=5)
    self.assertAlmostEqual(results['seg'][1], 4.66667, places=5)

  def testWMT20EnDeCSQMScores(self):
    evs = data.EvalSet('wmt20', 'en-de')
    results = {}
    for level in 'sys', 'doc', 'seg':
      scores = evs.Scores(level, 'csqm')
      if scores:
        n = len(scores)
        self.assertEqual(n, 10)
        results[level] = (scores['OPPO.1535'][0], scores['OPPO.1535'][-1])
    if results:
      self.assertAlmostEqual(results['sys'][0], 5.02116, places=5)
      self.assertAlmostEqual(results['sys'][1], 5.02116, places=5)
      self.assertAlmostEqual(results['doc'][0], 4.71795, places=5)
      self.assertAlmostEqual(results['doc'][1], 5.66667, places=5)
      self.assertAlmostEqual(results['seg'][0], 5.00000, places=5)
      self.assertAlmostEqual(results['seg'][1], 6.00000, places=5)

  def testWMT20SentBLEUSysScores(self):
    # All sentBLEU results from tables 5 and 6 in
    # www.statmt.org/wmt20/pdf/2020.wmt-1.77.pdf. (full, no-outlier)
    expected = {
        'cs-en': (0.844, 0.800),
        'de-en': (0.978, 0.786),
        'en-cs': (0.840, 0.436),
        'en-de': (0.934, 0.823),
        'en-iu': (0.129, 0.047),
        'en-ja': (0.946, 0.976),
        'en-pl': (0.950, 0.772),
        'en-ru': (0.981, 0.981),
        'en-ta': (0.881, 0.852),
        'en-zh': (0.927, 0.927),
        'iu-en': (0.649, 0.469),
        'ja-en': (0.974, 0.851),
        'km-en': (0.969, 0.969),
        'pl-en': (0.502, 0.284),
        'ps-en': (0.888, 0.888),
        'ru-en': (0.916, 0.833),
        'ta-en': (0.925, 0.829),
        'zh-en': (0.948, 0.950),
    }
    for lp in meta_info.DATA['wmt20']:
      evs = data.EvalSet('wmt20', lp, True)
      all_sys = evs.sys_names - evs.human_sys_names
      gold_scores = evs.Scores('sys', evs.StdHumanScoreName('sys'))
      sent_bleu = evs.Scores('sys', 'sentBLEU-ref')
      pearson_full = evs.Correlation(gold_scores, sent_bleu, all_sys).Pearson()
      pearson_no_outlier = evs.Correlation(
          gold_scores, sent_bleu, all_sys - evs.outlier_sys_names).Pearson()
      self.assertAlmostEqual(pearson_full[0], expected[lp][0], places=3)
      self.assertAlmostEqual(pearson_no_outlier[0], expected[lp][1], places=3)

  def testWMT20SentBLEUSegScores(self):
    # All sentBLEU results from tables 9 and 10 in
    # www.statmt.org/wmt20/pdf/2020.wmt-1.77.pdf  (full, no-outlier)
    expected = {
        'cs-en': (0.068, 0.057),
        'de-en': (0.413, -0.025),
        'en-cs': (0.432, 0.194),
        'en-de': (0.303, 0.155),
        'en-iu': (0.206, -0.084),
        'en-ja': (0.480, 0.390),  # Numbers in the table don't include outliers.
        'en-pl': (0.153, 0.067),
        'en-ru': (0.051, 0.051),
        'en-ta': (0.398, 0.206),
        'en-zh': (0.396, 0.396),
        'iu-en': (0.182, 0.170),
        'ja-en': (0.188, 0.061),
        'km-en': (0.226, 0.226),
        'pl-en': (-0.024, -0.046),
        'ps-en': (0.096, 0.096),
        'ru-en': (-0.005, -0.038),
        'ta-en': (0.162, 0.069),
        'zh-en': (0.093, 0.060),
    }
    for lp in meta_info.DATA['wmt20']:
      evs = data.EvalSet('wmt20', lp, True)
      all_sys = evs.sys_names - evs.human_sys_names
      gold_scores = evs.Scores('seg', evs.StdHumanScoreName('seg'))
      sent_bleu = evs.Scores('seg', 'sentBLEU-ref')
      kendall_full = evs.Correlation(
          gold_scores, sent_bleu, all_sys).KendallLike()
      kendall_no_outlier = evs.Correlation(
          gold_scores, sent_bleu, all_sys - evs.outlier_sys_names).KendallLike()
      self.assertAlmostEqual(kendall_full[0], expected[lp][0], places=3)
      self.assertAlmostEqual(kendall_no_outlier[0], expected[lp][1], places=3)

  def testWMT19BLEUSysScores(self):
    # All sys-level BLEU results from tables 3, 4, 5, in
    # https://www.aclweb.org/anthology/W19-5302.pdf
    expected = {
        'de-cs': 0.941,
        'de-en': 0.849,
        'de-fr': 0.891,
        'en-cs': 0.897,
        'en-de': 0.921,
        'en-fi': 0.969,
        'en-gu': 0.737,
        'en-kk': 0.852,
        'en-lt': 0.989,
        'en-ru': 0.986,
        'en-zh': 0.901,
        'fi-en': 0.982,
        'fr-de': 0.864,
        'gu-en': 0.834,
        'kk-en': 0.946,
        'lt-en': 0.961,
        'ru-en': 0.879,
        'zh-en': 0.899,
    }
    for lp in meta_info.DATA['wmt19']:
      evs = data.EvalSet('wmt19', lp, True)
      gold_scores = evs.Scores('sys', evs.StdHumanScoreName('sys'))
      bleu_scores = evs.Scores('sys', 'BLEU-ref')
      # Need to filter here because not all lps have wmt-z score for all
      # systems.
      sys_names = self._std_sys_names(evs).intersection(gold_scores)
      pearson = evs.Correlation(gold_scores, bleu_scores, sys_names).Pearson()
      self.assertAlmostEqual(pearson[0], expected[lp], places=3)

  def testWMT19BEERSegScores(self):
    # All seg-level BEER results from tables 6, 7, 8 in
    # https://www.aclweb.org/anthology/W19-5302.pdf
    expected = {
        'de-cs': 0.337,
        'de-en': 0.128,
        'de-fr': 0.293,
        'en-cs': 0.443,
        'en-de': 0.316,
        'en-fi': 0.514,
        'en-gu': 0.537,
        'en-kk': 0.516,
        'en-lt': 0.441,
        'en-ru': 0.542,
        'en-zh': 0.232,
        'fi-en': 0.283,
        'fr-de': 0.265,
        'gu-en': 0.260,
        'kk-en': 0.421,
        'lt-en': 0.315,
        'ru-en': 0.189,
        'zh-en': 0.371,
    }
    for lp in meta_info.DATA['wmt19']:
      evs = data.EvalSet('wmt19', lp, True)
      gold_scores = evs.Scores('seg', evs.StdHumanScoreName('seg'))
      beer_scores = evs.Scores('seg', 'BEER-ref')
      # Need to filter here because not all lps have wmt-z score for all
      # systems.
      sys_names = self._std_sys_names(evs).intersection(gold_scores)
      kl = evs.Correlation(gold_scores, beer_scores, sys_names).KendallLike()
      self.assertAlmostEqual(kl[0], expected[lp], places=3)


class DataTest(unittest.TestCase):

  def testAssignRanks(self):

    sig_matrix = np.array([
        0, 1, 1, 1,
        0, 0, 1, 1,
        0, 0, 0, 1,
        0, 0, 0, 0]).reshape((4, 4))
    self.assertEqual(data.AssignRanks(sig_matrix, 0.5), [1, 1, 1, 1])
    self.assertEqual(data.AssignRanks(sig_matrix, 1.0), [1, 2, 3, 4])

    sig_matrix = np.array([
        0, 1, 0, 1,
        0, 0, 1, 1,
        0, 0, 0, 1,
        0, 0, 0, 0]).reshape((4, 4))
    self.assertEqual(data.AssignRanks(sig_matrix, 0.5), [1, 1, 2, 2])

    sig_matrix = np.array([
        0, 0, 1, 1,
        0, 0, 1, 1,
        0, 0, 0, 0,
        0, 0, 0, 0]).reshape((4, 4))
    self.assertEqual(data.AssignRanks(sig_matrix, 0.5), [1, 2, 2, 3])

  def testMapPositions(self):
    items = ['aa', 'aa', 'bb', 'bb', 'bb', 'aa']
    d = data._MapPositions(items)
    self.assertEqual(d['aa'], [[0, 2], [5, 6]])
    self.assertEqual(d['bb'], [[2, 5]])
    self.assertEqual(data._UnmapPositions(d), items)

    items = ['aa', 'aa', 'bb', 'cc', 'cc', 'cc']
    d = data._MapPositions(items, True)
    self.assertEqual(d['aa'], [0, 2])
    self.assertEqual(d['bb'], [2, 3])
    self.assertEqual(d['cc'], [3, 6])
    self.assertEqual(data._UnmapPositions(d, True), items)


if __name__ == '__main__':
  unittest.main()
