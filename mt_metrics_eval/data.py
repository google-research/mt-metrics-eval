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
"""Access to standard datasets."""

import collections
import copy
import os
import tarfile
from typing import Any, Callable, Dict, Iterable, List, Sequence, Set
import urllib.request
from mt_metrics_eval import meta_info
from mt_metrics_eval import stats
import numpy as np
import glob



TGZ = 'https://storage.googleapis.com/mt-metrics-eval/mt-metrics-eval-v2.tgz'


class EvalSet:
  """Data for a test set and one language pair."""

  def __init__(self, name: str,
               lp: str,
               read_stored_metric_scores: bool = False,
               info: meta_info.MetaInfo = None,
               path: str = None,
               strict: bool = False):
    """Load data for a given test set and language pair.

    By default this will load meta-info and read data for one of the known
    test-set/language-pair combinations in meta_info.DATA. To use alternative
    meta-info or data, pass info and path arguments.

    Args:
      name: Name of test set. If info is None, this must match a top-level key
        in meta_info.DATA.
      lp: Language pair, eg 'en-de'. If info is None, this must match a key in
        meta_info.DATA[name]
      read_stored_metric_scores: Read stored scores for automatic metrics for
        this dataset. This makes loading slower, and is only needed for
        analyzing or directly comparing to these scores.
      info: Optional meta info for this set.
      path: Optional path to parent directory: path/name should contain data
        structured as described under 'File organization and naming convention'
        in the README.
      strict: If False, score files that are missing all entries for some
        systems will be 'repaired' by silently adding 0 scores for those systems
        instead of raising an exception.
    """
    self.name = name
    self.lp = lp

    if info is None:
      if name not in meta_info.DATA:
        raise ValueError('Unknown dataset: %s' % name)
      if lp not in meta_info.DATA[name]:
        raise ValueError('Language pair not in %s: %s' % (name, lp))
      info = meta_info.DATA[name][lp]
    self.info = copy.deepcopy(info)
    self._std_human_scores = {}
    self._std_human_scores['sys'] = self.info.std_gold_sys
    self._std_human_scores['doc'] = self.info.std_gold_doc
    self._std_human_scores['seg'] = self.info.std_gold_seg

    self._ReadDataset(name, lp, read_stored_metric_scores, path, strict)

    # Check compatibility between info and data read in.
    # No checks for primary metrics because there are no hard requirements:
    # no metrics for this lp need to be primary.
    if self.std_ref not in self.ref_names:
      raise ValueError(f'Standard ref {self.std_ref} not in known set.')
    for level in ['sys', 'doc', 'seg']:
      gold = self.StdHumanScoreName(level)
      if gold and gold not in self.human_score_names:
        raise ValueError(
            f'Standard {level=} gold score {gold} not in known set.')
    if not self.outlier_sys_names.issubset(self.sys_names):
      raise ValueError(
          f'Outlier systems {self.outlier_sys_names - self.sys_names} '
          'not in known set.')

  @property
  def doc_names(self) -> Sequence[str]:
    """Names of documents, in order."""
    return self._docs.keys()

  @property
  def ref_names(self) -> Set[str]:
    """Names of available references."""
    return set(self._all_refs.keys())

  @property
  def std_ref(self) -> str:
    """Name of standard reference."""
    return self.info.std_ref

  @property
  def sys_names(self) -> Set[str]:
    """Names of all 'systems' for which output is available."""
    return set(self._sys_outputs.keys())

  @property
  def human_sys_names(self) -> Set[str]:
    """Names of systems in sys_names that are human output."""
    return self._human_sys_names

  @property
  def outlier_sys_names(self) -> Set[str]:
    """Names of systems in sys_names considered to be outliers."""
    return self.info.outlier_systems

  @property
  def human_score_names(self) -> Set[str]:
    """Names of different human scores available, eg 'wmt-z', 'mqm'."""
    return self._human_score_names

  def StdHumanScoreName(self, level) -> str:
    """Name of standard human score for a given level, empty string if None."""
    return self._std_human_scores[level]

  @property
  def metric_names(self) -> Set[str]:
    """Full names of different metrics available."""
    return self._metric_names

  @property
  def metric_basenames(self) -> Set[str]:
    """Basenames of different metrics available."""
    return self._metric_basenames

  @property
  def primary_metrics(self) -> Set[str]:
    """Base names of primary metrics, empty set if none."""
    return self.info.primary_metrics

  def BaseMetric(self, metric_name: str) -> str:
    """Base name for a given metric."""
    return self.ParseMetricName(metric_name)[0]

  def ReferencesUsed(self, metric_name: str) -> Set[str]:
    """Reference(s) used by a metric."""
    return self.ParseMetricName(metric_name)[1]

  @property
  def docs(self) -> Dict[str, List[int]]:
    """Map from doc name to [beg, end+1] segment positions."""
    return self._docs

  @property
  def src(self) -> List[str]:
    """Segments in the source text, in order."""
    return self._src

  @property
  def all_refs(self) -> Dict[str, List[str]]:
    """Map from reference name to text for that reference."""
    return self._all_refs

  @property
  def sys_outputs(self) -> Dict[str, List[str]]:
    """Map from system name to output text from that system."""
    return self._sys_outputs

  def Scores(self, level: str, scorer: str) -> Dict[str, List[float]]:
    """Get stored scores assigned to text units at a given level.

    Args:
      level: Text units to which scores apply, one of 'sys', 'doc', 'seg'.
      scorer: Method used to produce scores, may be any string in
        human_score_names or metric_names.

    Returns:
      Mapping from system names to lists of float scores, or None if scores
      aren't available at this level (eg BLEU at segment level). If level is
      'sys', the lists contain one element, otherwise elements corresponding to
      documents or segments in order. Some entries in each list may be None if
      these are human scores.
    """
    if level in self._scores and scorer in self._scores[level]:
      return self._scores[level][scorer]
    else:
      return None

  def Correlation(self,
                  gold_scores: Dict[str, List[float]],
                  metric_scores: Dict[str, List[float]],
                  sys_names: Iterable[str] = None):
    """Get correlation statistics for given metric scores.

    Args:
      gold_scores: Gold scores to use, same format as metric_scores, except that
        score lists may contain None values.
      metric_scores: Metric scores to evaluate, a map from system names to lists
        of float scores.
      sys_names: Names of systems to use in comparison, must exist in both
        metric_scores and gold_scores. Default is to use all systems for which
        gold scores are available.

    Returns:
      A stats.Correlation object for computing correlation statistics.
    """
    if sys_names is None:
      sys_names = gold_scores
    sys_names = set(sys_names)
    if not sys_names.issubset(metric_scores):
      raise ValueError(
          f'Missing metric scores: {sys_names - set(metric_scores)}')
    if not sys_names.issubset(gold_scores):
      raise ValueError(f'Missing gold scores: {sys_names - set(gold_scores)}')

    all_gold_scores, all_metric_scores = [], []
    for sys_name in sys_names:
      gscores, mscores = gold_scores[sys_name], metric_scores[sys_name]
      if len(gscores) != len(mscores):
        raise ValueError('Wrong number of scores for system %s: %d vs %d' %
                         (sys_name, len(gscores), len(mscores)))
      all_gold_scores.extend(gscores)
      all_metric_scores.extend(mscores)
    return stats.Correlation(len(sys_names), all_gold_scores, all_metric_scores)

  def ParseHumanScoreFilename(self, filename):
    """Parse a human-score filename into lang, name, and level components."""
    # SRC-TGT.NAME.LEVEL.score
    toks = os.path.basename(filename).split('.')
    if len(toks) < 4 or toks[-1] != 'score':
      raise ValueError(f'Bad format for human scores: {filename}')
    lp = toks[0]
    name = '.'.join(toks[1:-2])
    level = toks[-2]
    return lp, name, level

  def ParseMetricFilename(self, filename):
    """Parse a metric filename into name and level components."""
    # NAME.LEVEL.score
    name, level, exten = os.path.basename(filename).rsplit('.', maxsplit=2)
    if exten != 'score' or level not in ['sys', 'doc', 'seg']:
      raise ValueError(
          f'Metric file {filename} not in NAME-REF.LEVEL.score format.')
    return name, level

  def ParseMetricName(self, metric_name):
    """Parse a metric name into basename and reference(s) used."""
    # BASENAME-REF
    if '-' not in metric_name:
      raise ValueError(
          f'Metric {metric_name} not in NAME-REF format')
    basename, refs = metric_name.rsplit('-', maxsplit=1)
    if refs == 'all':
      refset = set(self.ref_names)
    elif refs == 'src':
      refset = set()
    else:
      refset = set(refs.split('.'))
    return basename, refset

  def CheckScores(self, scores_map, scorer_name, level, human, repair=False):
    """Check and optionally repair scores returned by ReadScoreFile.

    Repairs are limited to metric (not human) scores, and involve adding 0
    scores for systems that are missing from scores_map. If repair is True,
    these errors don't raise an exception. All other errors always cause an
    exception.

    Args:
      scores_map: Map from system-names to scores as returned by ReadScoreFile.
      scorer_name: Name of this scorer, either a metric or human score.
      level: Granularity, one of 'sys', 'doc', or 'seg'.
      human: True for human scores, False for metric scores.
      repair: If True and if human is False, add 0s for missing systems.

    Returns:
      List of systems that got 0-padded during repair.
    """
    expected_len = {'sys': 1, 'doc': len(self.docs), 'seg': len(self.src)}
    for sys_name, scores in scores_map.items():
      if sys_name not in self.sys_names:
        raise ValueError(
            f'Unknown system in {scorer_name}.{level} scores: {sys_name}.')
      if len(scores) != expected_len[level]:
        raise ValueError(
            f'{scorer_name}.{level} contains wrong number of scores: '
            f'{len(scores)} vs {expected_len[level]}')
      if not human and None in scores:
        raise ValueError(f'{scorer_name}.{level} contains None elements.')

    added_scores = []
    if not human:
      refs_used = self.ReferencesUsed(scorer_name)
      for sys_name in self.sys_names:
        if sys_name in scores_map or sys_name in refs_used:
          continue
        if repair:
          scores_map[sys_name] = [0] * expected_len[level]
          added_scores.append(sys_name)
        else:
          raise ValueError(
              f'{scorer_name}.{level} is missing required scores for system '
              f'{sys_name}')
    return added_scores

  def _ReadDataset(self, name, lp, read_stored_metric_scores, path, strict):
    """Read data for given name and language pair."""

    if not path:
      path = LocalDir(root_only=False)
      if not os.path.exists(path):
        raise ValueError('%s not found. Run mtme --download.' % path)

    d = os.path.join(path, name)
    self._docs = ReadDocPositions(os.path.join(d, 'documents', '%s.docs' % lp))
    self._src = ReadTextFile(os.path.join(d, 'sources', '%s.txt' % lp))

    self._all_refs = {}
    for filename in glob.glob(os.path.join(d, 'references', '%s.*.txt' % lp)):
      refname = filename.split('.')[-2]
      if '-' in refname or refname in ['all', 'src']:
        assert False, f'Invalid reference name: {refname}'
      self._all_refs[refname] = ReadTextFile(filename)

    self._outlier_sys_names, self._human_sys_names = set(), set()
    self._sys_outputs = {}
    for filename in glob.glob(os.path.join(d, 'system-outputs', lp, '*.txt')):
      sysname = os.path.basename(filename)[:-len('.txt')]
      self._sys_outputs[sysname] = ReadTextFile(filename)
      if sysname in self._all_refs:
        self._human_sys_names.add(sysname)

    self._human_score_names = set()
    self._scores = {'sys': {}, 'doc': {}, 'seg': {}}
    for filename in glob.glob(
        os.path.join(d, 'human-scores', '%s.*.score' % lp)):
      lp, scorer, level = self.ParseHumanScoreFilename(
          os.path.basename(filename))
      self._human_score_names.add(scorer)
      assert scorer not in self._scores[level], scorer
      self._scores[level][scorer] = ReadScoreFile(filename)

    self._metric_names = set()
    self._metric_basenames = set()
    if read_stored_metric_scores:
      for filename in glob.glob(
          os.path.join(d, 'metric-scores', lp, '*.score')):
        scorer, level = self.ParseMetricFilename(filename)
        assert scorer not in self._scores[level]
        assert self.ReferencesUsed(scorer).issubset(self.ref_names)
        self._metric_names.add(scorer)
        self._metric_basenames.add(self.BaseMetric(scorer))
        self._scores[level][scorer] = ReadScoreFile(filename)

    # Check contents
    for txt in self.all_refs.values():
      assert len(txt) == len(self.src), f'Bad length for reference {txt}'
    for txt in self.sys_outputs.values():
      assert len(txt) == len(self.src), f'Bad length for output {txt}'
    for level in 'sys', 'doc', 'seg':
      if level in self._scores:
        for scorer_name, scores_map in self._scores[level].items():
          self.CheckScores(scores_map, scorer_name, level,
                           scorer_name in self.human_score_names,
                           repair=not strict)


def LocalDir(root_only=True):
  """Location for local dir: $HOME/.mt-metrics-eval."""
  path = os.path.join(os.path.expanduser('~'), '.mt-metrics-eval')
  if not root_only:
    path = os.path.join(path, 'mt-metrics-eval-v2')
  return path


def _CopyTgz(dest):
  with urllib.request.urlopen(TGZ) as f, open(dest, 'wb') as out:
    out.write(f.read())


def Download():
  """Download database into LocalDir()."""
  path = LocalDir()
  os.makedirs(path, exist_ok=True)
  local_tgz = os.path.join(path, os.path.basename(TGZ))
  _CopyTgz(local_tgz)
  with tarfile.open(local_tgz, 'r:*') as tar:
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(tar, path)


def ReadDocPositions(filename):
  """Read docs file and return map from docname to [beg, end+1] positions."""
  docs = {}
  with open(filename) as f:
    for i, line in enumerate(f):
      _, doc = line.split()
      if doc not in docs:
        docs[doc] = [i, i]
      docs[doc][1] += 1
  return docs


def ReadTextFile(filename):
  with open(filename) as f:
    lines = [line.rstrip() for line in f]
  return lines


def ReadScoreFile(filename):
  scores = collections.defaultdict(list)  # sys -> [scores]
  with open(filename) as f:
    for line in f:
      sysname, score = line.split()
      scores[sysname].append(float(score) if score != 'None' else None)
  return scores


def GetCorrelations(evs: EvalSet, level: str, main_refs: set[str],
                    close_refs: set[str], include_human: bool,
                    include_outliers: bool, gold_name: str,
                    primary_metrics: bool) -> dict[str, stats.Correlation]:
  """Convenience function to generate sufficient stats for given parameters.

  Args:
    evs: EvalSet to use.
    level: Granularity, one of 'sys', 'doc', or 'seg'. Not all granularities
      are available for all eval sets. Check that evs.StdHumanScoreName(level)
      is not the empty string.
    main_refs: Set of references to use. Metric variants that use references
      outside this set are excluded, as are human outputs that match any of
      these references. Normally this will be the single standard reference,
      evs.std_ref.
    close_refs: Any references that are known to be close to main_refs, for
      example post-edited versions. Any human outputs that match these
      references will be excluded.
    include_human: Include any available human outputs, subject to the above
      constraints.
    include_outliers: Include any systems considered to be outliers.
    gold_name: Name of human gold scores to use, any value in
      evs.human_score_names. Normally this will be evs.StdHumanScoreName(level).
    primary_metrics: Include primary metrics only.

  Returns:
    Map from metric names to Correlation objects from which correlation and
    significance statistics can be computed.
  """

  # List of systems to be scored.
  sys_names = evs.sys_names - main_refs - close_refs
  if not include_human:
    sys_names -= evs.human_sys_names
  if not include_outliers:
    sys_names -= evs.outlier_sys_names

  # Get gold scores and filter outputs to those for which we have gold scores.
  gold_scores = evs.Scores(level, gold_name)
  sys_names = sys_names.intersection(gold_scores)

  # Generate 'Correlation' objects for all specified metrics.
  correlations = {}  # metric -> Correlation
  for metric_name in evs.metric_names:
    base_name, metric_refs = evs.ParseMetricName(metric_name)
    if primary_metrics and base_name not in evs.primary_metrics:
      continue
    if not metric_refs.issubset(main_refs):
      continue
    metric_scores = evs.Scores(level, metric_name)
    if not metric_scores:  # Metric not available at this level.
      continue
    correlations[metric_name] = evs.Correlation(
        gold_scores, metric_scores, sys_names)

  return correlations


def CompareMetrics(
    metric_corrs: dict[str, stats.Correlation],
    corr_fcn: Callable[[list[float], list[float]], tuple[float, float]],
    average_by: str = 'none', k: int = 1000, pval: float = 0.05,
    ) -> tuple[dict[str, tuple[float, float]], Any]:
  """Compare a set of metrics using a given correlation function.

  This function uses a permutation test to compute significant differences
  between correlations; it can be very slow when correlation vectors are large,
  especially when averaging is used.

  Args:
    metric_corrs: Map from metric names to stats.Correlation objects containing
      metric and gold scores for the same data. This is the format returned by
      GetCorrelations().
    corr_fcn: Function for generating correlations from metric and gold score
      vectors. Correlations are assumed to be in the first element of the
      returned tuple (2nd element is ignored). Pass a function like
      scipy.stats.pearsonr rather than a stats.CorrFunction wrapper (the point
      of this function is to save you the trouble of creating the wrapper
      yourself).
    average_by: What to average over when computing final correlations:
      'none' - Treat all scores as single vectors, and compute a single
        correlation. This is the only option that makes sense for system-level
        scores.
      'item' - Average over correlations for vectors of scores for the same item
        across all systems.
      'sys' - Average over the correlations for vectors of scores for the same
        system across all items.
    k: Number of resampling runs for PermutationSigDiff test.
    pval: p-value for determining significant differences in ranks.

  Returns:
    1. Mapping from metric name to (correlation, rank) pairs, where rank is
       the rank of the metric's significance cluster. Keys are ordered by
       decreasing correlation.
    2. Significance matrix: a square numpy array whose rows and columns
       represent metrics, sorted to match the keys in the returned correlation
       map. sig_matrix[i, j] contains the p-value for the null hypothesis that
       the correlation for metric j is >= the correlation for metric i. Only
       entries for which j > i are valid.
  """
  assert metric_corrs
  assert average_by in ('none', 'item', 'sys'), 'Bad average_by value.'

  first_corr = list(metric_corrs.values())[0]
  corr_wrapper = stats.CorrFunction(
      corr_fcn,
      num_sys=0 if average_by == 'none' else first_corr.num_sys,
      filter_nones=first_corr.none_count,
      by_system=average_by == 'sys')

  # Compute metric correlations, ordered by decreasing correlation.
  corrs_and_ranks = {}
  for m, c in metric_corrs.items():
    corrs_and_ranks[m] = [corr_wrapper(c.gold_scores, c.metric_scores)[0], 0]
  corrs_and_ranks = dict(
      sorted(corrs_and_ranks.items(), key=lambda x: -x[1][0]))

  # Compute top half of significance matrix.
  n = len(corrs_and_ranks)
  sig_matrix = np.zeros((n, n))
  for i in range(n):
    m1 = list(corrs_and_ranks)[i]
    for j in range(i + 1, n):
      m2 = list(corrs_and_ranks)[j]
      sig_matrix[i, j] = stats.PermutationSigDiff(
          metric_corrs[m2], metric_corrs[m1], corr_wrapper, k)

  # Determine ranks.
  ranks = AssignRanks(sig_matrix, pval)
  for i, m in enumerate(corrs_and_ranks):
    corrs_and_ranks[m][1] = ranks[i]

  return corrs_and_ranks, sig_matrix


def AssignRanks(sig_matrix, pval):
  """Assign ranks to metrics from a pairwise significance matrix.

  Args:
    sig_matrix: Upper-diagonal square numpy array whose rows and columns
      represent metrics, sorted in order of decreasing correlation.
      sig_matrix[i, j] contains the p-value for the null hypothesis that the
      correlation for metric j is >= the correlation for metric i.
    pval: Maximum p-value threshold for assigning significance.

  Returns:
    List whose ith value contains the rank for the ith metric. Metrics assigned
    rank r are those not significantly outperformed by any other metric of rank
    r, nor outperformed by any metric of rank < r.
  """
  assert sig_matrix.ndim == 2
  assert sig_matrix.shape[0] == sig_matrix.shape[1]
  n = sig_matrix.shape[0]
  ranks = [0] * n
  current_rank = 1
  start_index = 0
  for i in range(n):
    if any(sig_matrix[:, i][start_index: i] <= pval):
      current_rank += 1
      start_index = i
    ranks[i] = current_rank
  return ranks
