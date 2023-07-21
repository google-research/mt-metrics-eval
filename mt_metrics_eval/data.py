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
import itertools
import os
import sys
import tarfile
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple
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
        in the README. You may use multiple paths in a form of a list.
        In this case, the first path must contain the data for the testsets.
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
    self._std_human_scores = self.info.std_gold

    self._ReadDataset(name, lp, read_stored_metric_scores, path, strict)

    # Check compatibility between info and data read in.
    # No checks for primary metrics because there are no hard requirements:
    # no metrics for this lp need to be primary.
    if self.std_ref not in self.ref_names:
      raise ValueError(f'Standard ref {self.std_ref} not in known set.')
    for level in self._std_human_scores:
      gold = self._std_human_scores[level]
      if level not in self._scores or gold not in self._scores[level]:
        raise ValueError(
            f'Standard {level=} gold scores {gold} not in known set.')
    if not self.outlier_sys_names.issubset(self.sys_names):
      raise ValueError(
          f'Outlier systems {self.outlier_sys_names - self.sys_names} '
          'not in known set.')

  @property
  def levels(self) -> Set[str]:
    """Levels for which scores exist, subset of {sys, domain, doc, seg}."""
    return set(self._scores.keys())

  @property
  def domain_names(self) -> Sequence[str]:
    """Names of domains, in canonical order."""
    return self._domains.keys()

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
    """Name of standard human score for a given level in self.levels."""
    if level in self._std_human_scores:
      return self._std_human_scores[level]
    else:
      return None

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

  def DisplayName(self, metric_name: str, fmt='spreadsheet') -> str:
    """Display name for a given metric."""
    name, refs = self.ParseMetricName(metric_name)
    if fmt == 'spreadsheet':
      if name not in self.primary_metrics:
        name = '*' + name
      if not refs:
        name = name + '[noref]'
    elif fmt == 'latex':
      if not refs:
        name = name + '*'
      if name in self.primary_metrics:
        name = f'\textbf{{{name}}}'
    else:
      raise ValueError(f'Unkown format: {fmt}')
    return name

  def ReferencesUsed(self, metric_name: str) -> Set[str]:
    """Reference(s) used by a metric."""
    return self.ParseMetricName(metric_name)[1]

  @property
  def domains(self) -> Dict[str, List[List[int]]]:
    """Map from domain name to [[beg, end+1], ...] segment position lists."""
    return self._domains

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
      level: Text units to which scores apply, one of 'sys', 'domain', 'doc', or
        'seg'.
      scorer: Method used to produce scores, may be any string in
        human_score_names or metric_names.

    Returns:
      Mapping from system names to lists of float scores, or None if scores
      aren't available at this level (eg BLEU at segment level). If level is
      'sys', the lists contain one element, otherwise elements corresponding to
      domains, documents or segments in order. Some entries in each list may be
      None if these are human scores.
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
    if exten != 'score' or level not in ['sys', 'domain', 'doc', 'seg']:
      raise ValueError(
          f'Metric file {filename} not in NAME-REF.LEVEL.score format.')
    return name, level

  def ParseMetricName(self, metric_name):
    """Parse a metric name into basename and reference(s) used."""
    # BASENAME-REF
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
      level: Granularity, one of 'sys', 'domain', 'doc', or 'seg'.
      human: True for human scores, False for metric scores.
      repair: If True and if human is False, add 0s for missing systems.

    Returns:
      List of systems that got 0-padded during repair.
    """
    expected_len = {'sys': 1, 'domain': len(self.domains),
                    'doc': len(self.docs), 'seg': len(self.src)}
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

  def DocsPerSeg(self):
    """Return a list of document names for each segment, in order."""
    return _UnmapPositions(self.docs, contiguous=True)

  def DomainsPerSeg(self):
    """Return a list of domain names for each segment, in order."""
    return _UnmapPositions(self.domains, contiguous=False)

  def _ReadDataset(self, name, lp, read_stored_metric_scores, path, strict):
    """Read data for given name and language pair."""

    if path is None:
      path = LocalDir(root_only=False)
      if not os.path.exists(path):
        raise ValueError('%s not found. Run mtme --download.' % path)

    if isinstance(path, list):
      metric_scores_paths = path
      path = path[0]  # Use first path for dataset resource files.
    else:
      metric_scores_paths = [path]

    d = os.path.join(path, name)
    doc_lines = _ReadTextFile(os.path.join(d, 'documents', '%s.docs' % lp))
    self._domains = _MapPositions([d.split()[0] for d in doc_lines])
    # Canonicalized domain order, since there is no natural order.
    self._domains = {k: self._domains[k] for k in sorted(self._domains)}
    self._docs = _MapPositions([d.split()[1] for d in doc_lines], True)
    self._src = _ReadTextFile(os.path.join(d, 'sources', '%s.txt' % lp))

    self._all_refs = {}
    for filename in glob.glob(os.path.join(d, 'references', '%s.*.txt' % lp)):
      refname = filename.split('.')[-2]
      if '-' in refname or refname in ['all', 'src']:
        assert False, f'Invalid reference name: {refname}'
      self._all_refs[refname] = _ReadTextFile(filename)

    self._outlier_sys_names, self._human_sys_names = set(), set()
    self._sys_outputs = {}
    for filename in glob.glob(os.path.join(d, 'system-outputs', lp, '*.txt')):
      sysname = os.path.basename(filename)[:-len('.txt')]
      self._sys_outputs[sysname] = _ReadTextFile(filename)
      if sysname in self._all_refs:
        self._human_sys_names.add(sysname)

    self._human_score_names = set()
    self._scores = {}
    for filename in glob.glob(
        os.path.join(d, 'human-scores', '%s.*.score' % lp)):
      lp, scorer, level = self.ParseHumanScoreFilename(
          os.path.basename(filename))
      self._human_score_names.add(scorer)
      if level not in self._scores:
        self._scores[level] = {}
      assert scorer not in self._scores[level], scorer
      if level == 'domain':
        self._scores[level][scorer] = ReadDomainScoreFile(
            filename, self.domain_names)
      else:
        self._scores[level][scorer] = ReadScoreFile(filename)

    self._metric_names = set()
    self._metric_basenames = set()
    if read_stored_metric_scores:
      for md in metric_scores_paths:
        md = os.path.join(md, name)
        for filename in glob.glob(
            os.path.join(md, 'metric-scores', lp, '*.score')):
          scorer, level = self.ParseMetricFilename(filename)
          if level not in self._scores:
            self._scores[level] = {}
          assert scorer not in self._scores[level]
          assert self.ReferencesUsed(scorer).issubset(self.ref_names)
          self._metric_names.add(scorer)
          self._metric_basenames.add(self.BaseMetric(scorer))
          if level == 'domain':
            self._scores[level][scorer] = ReadDomainScoreFile(
                filename, self.domain_names)
          else:
            self._scores[level][scorer] = ReadScoreFile(filename)

    # Check contents
    for txt in self.all_refs.values():
      assert len(txt) == len(self.src), f'Bad length for reference {txt}'
    for txt in self.sys_outputs.values():
      assert len(txt) == len(self.src), f'Bad length for output {txt}'
    for level in self._scores:
      for scorer_name, scores_map in self._scores[level].items():
        self.CheckScores(scores_map, scorer_name, level,
                         scorer_name in self.human_score_names,
                         repair=not strict)


def _MapPositions(item_list, contiguous=False):
  """Map a list of items to position(s) of occurences.

  Args:
    item_list: List of arbitrary items (eg document or domain names).
    contiguous: Duplicate items in list must be contiguous.

  Returns:
    Dict mapping items to positions. If contiguous is set, dict values are
    pairs of [beg, end+1] positions, otherwise they are lists of such pairs.
  """
  item_dict = {}
  pos = 0
  for k, g in itertools.groupby(list(item_list)):
    end = pos + len(list(g))
    if contiguous:
      assert k not in item_dict, f'Non-contiguous occurrences of {k}'
      item_dict[k] = [pos, end]
    else:
      if k not in item_dict:
        item_dict[k] = []
      item_dict[k].append([pos, end])
    pos = end
  return item_dict


def _UnmapPositions(item_dict, contiguous=False):
  """Reverse _MapPositions() and return original list."""
  if contiguous:
    item_dict = {d: [v] for d, v in item_dict.items()}
  maxlen = 0
  for v in item_dict.values():
    maxlen = max(maxlen, max(p[1] for p in v))
  item_list = [None] * maxlen
  for k, v in item_dict.items():
    for p in v:
      item_list[slice(*p)] = [k] * (p[1] - p[0])
  return item_list


def _CopyTgz(dest):
  with urllib.request.urlopen(TGZ) as f, open(dest, 'wb') as out:
    out.write(f.read())


def _ReadTextFile(filename):
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


def ReadDomainScoreFile(filename, ordered_domains):
  """Read a domain score file, return a map with correctly-ordered scores."""
  scores = {}  # sys -> [scores]
  domain_to_index = {d: i for i, d in enumerate(ordered_domains)}
  with open(filename) as f:
    for line in f:
      domain, sysname, score = line.split()
      score = float(score) if score != 'None' else None
      if sysname not in scores:
        scores[sysname] = [None] * len(ordered_domains)
      scores[sysname][domain_to_index[domain]] = score
  return scores


def LocalDir(root_only=True):
  """Location for local dir: $HOME/.mt-metrics-eval."""
  path = os.path.join(os.path.expanduser('~'), '.mt-metrics-eval')
  if not root_only:
    path = os.path.join(path, 'mt-metrics-eval-v2')
  return path


def Download():
  """Download database into LocalDir()."""
  path = LocalDir()
  os.makedirs(path, exist_ok=True)
  local_tgz = os.path.join(path, os.path.basename(TGZ))
  _CopyTgz(local_tgz)
  with tarfile.open(local_tgz, 'r:*') as tar:
    tar.extractall(path)


def GetCorrelations(evs: EvalSet, level: str, main_refs: Set[str],
                    close_refs: Set[str], include_human: bool,
                    include_outliers: bool, gold_name: str,
                    primary_metrics: bool, domain: str = None,
                    extern_metrics: Dict[str, Dict[str, List[float]]] = None,
                    ) -> Dict[str, stats.Correlation]:
  """Convenience function to generate sufficient stats for given parameters.

  Note that this doesn't actually compute correlations, it only extracts the
  vectors over which correlations can later be computed, for a desired set of
  metrics.

  Args:
    evs: EvalSet to use.
    level: Granularity, one of the values in evs.levels.
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
      evs.human_score_names, or the special value 'std' to use
      evs.StdHumanScoreName(level).
    primary_metrics: Include primary metrics only.
    domain: If not None, must be a value in evs.domain_names; this indicates
      that only the scores pertaining to that domain should be used. In this
      case, if level is 'sys', it is treated as if it were 'domain'.
    extern_metrics: A dict containing metrics not in evs. Each entry should
      map a correctly-formatted metric name to a dict containing scores for
      all systems, at the correct granularity.

  Returns:
    Map from metric names to stats.Correlation objects from which correlation
    and significance statistics can be computed.
  """
  if domain is not None:
    assert domain in evs.domain_names
    if level == 'sys':
      level = 'domain'
  assert level in evs.levels

  def _Filter(scores):
    if domain is None:
      return scores
    new_scores = {}
    mask = {
        'domain': [d == domain for d in evs.domain_names],
        'doc': [d == domain for d in evs.DocsPerSeg()],
        'seg': [d == domain for d in evs.DomainsPerSeg()]
        }[level]
    for k, vals in scores.items():
      assert len(vals) == len(mask)
      new_scores[k] = [s for s, m in zip(vals, mask) if m]
    return new_scores

  # List of systems to be scored.
  sys_names = evs.sys_names - main_refs - close_refs
  if not include_human:
    sys_names -= evs.human_sys_names
  if not include_outliers:
    sys_names -= evs.outlier_sys_names

  # Get gold scores and filter outputs to those for which we have gold scores.
  if gold_name == 'std':
    gold_name = evs.StdHumanScoreName(level)
  gold_scores = _Filter(evs.Scores(level, gold_name))
  sys_names = sys_names.intersection(gold_scores)

  # Generate 'Correlation' objects for all specified metrics.
  correlations = {}  # metric -> Correlation
  for metric_name in evs.metric_names:
    base_name, metric_refs = evs.ParseMetricName(metric_name)
    if (primary_metrics and evs.primary_metrics and
        base_name not in evs.primary_metrics):
      continue
    if not metric_refs.issubset(main_refs):
      continue
    metric_scores = evs.Scores(level, metric_name)
    if not metric_scores:  # Metric not available at this level.
      continue
    correlations[metric_name] = evs.Correlation(
        gold_scores, _Filter(metric_scores), sys_names)

    # Add in extra metrics.
    if extern_metrics is not None:
      for metric_name, scores in extern_metrics.items():
        correlations[metric_name] = evs.Correlation(
            gold_scores, _Filter(scores), sys_names)

  return correlations


def CompareMetrics(
    metric_corrs: Dict[str, stats.Correlation],
    corr_fcn: Callable[[List[float], List[float], ...], Tuple[float, float]],
    average_by: str = 'none',
    k: int = 1000,
    psd: stats.PermutationSigDiffParams = stats.PermutationSigDiffParams(),
    pval: float = 0.05,
    replace_nans_with_zeros: bool = True,
    **corr_fcn_args,
    ) -> Tuple[Dict[str, Tuple[float, float]], np.ndarray]:
  """Compare a set of metrics using a given correlation function.

  This function uses a permutation test to compute significant differences
  between correlations; it can be very slow when correlation vectors are large,
  especially when averaging is used. Set k=0 to speed it up if you only want to
  rank metrics without performing pairwise significance tests.

  Args:
    metric_corrs: Map from metric names to stats.Correlation objects containing
      metric and gold scores for the same data. This is the format returned by
      GetCorrelations().
    corr_fcn: Function for generating correlations from metric and gold score
      vectors. Correlations are assumed to be in the first element of the
      returned tuple (2nd element is ignored). Pass a function from scipy.stats
      or any of the plain correlation functions from stats (not wrapped in
      AverageCorrelation).
    average_by: What to average over when computing final correlations:
      'none' - Treat all scores as single vectors, and compute a single
        correlation. This is the only option that makes sense for system-level
        scores.
      'item' - Average over correlations for vectors of scores for the same item
        across all systems.
      'sys' - Average over the correlations for vectors of scores for the same
        system across all items.
    k: Number of resampling runs for PermutationSigDiff test. If k is 0, no
      test is performed, and only the raw ranks of metrics are returned, along
      with an empty significance matrix.
    psd: Additional parameters for stats.PermutationSigDiff.
    pval: p-value for determining significant differences in ranks.
    replace_nans_with_zeros: If True, replace NaN correlation values with 0.
      If False, remove these values from the computation. The former setting
      will penalize metrics that produce NaN values because they assign all
      items the same score.
    **corr_fcn_args: Optional extra arguments to corr_fcn.

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
  corr_wrapper = stats.AverageCorrelation(
      corr_fcn,
      first_corr.num_sys,
      average_by=average_by,
      filter_nones=first_corr.none_count,
      replace_nans_with_zeros=replace_nans_with_zeros,
      **corr_fcn_args)

  # Compute metric correlations, ordered by decreasing correlation.
  corrs_and_ranks = {}
  for m, c in metric_corrs.items():
    corrs_and_ranks[m] = [corr_wrapper(c.gold_scores, c.metric_scores)[0], 0]
  corrs_and_ranks = dict(
      sorted(corrs_and_ranks.items(), key=lambda x: -x[1][0]))

  # Compute significance matrix and determine ranks.
  sig_matrix = ComputeSigMatrix(
      metric_corrs, corrs_and_ranks, corr_fcn, average_by, k,
      psd, replace_nans_with_zeros, **corr_fcn_args)
  ranks = AssignRanks(sig_matrix, pval)
  for i, m in enumerate(corrs_and_ranks):
    corrs_and_ranks[m][1] = ranks[i]

  return corrs_and_ranks, sig_matrix


def CompareMetricsWithGlobalAccuracy(
    evs_list: List[EvalSet],
    main_refs_list: List[Set[str]],
    close_refs_list: List[Set[str]],
    include_human: bool,
    include_outliers: bool,
    gold_name: str,
    primary_metrics: bool,
    domain: str = None,
    k: int = 1000,
    psd: stats.PermutationSigDiffParams = stats.PermutationSigDiffParams(),
    pval: float = 0.05,
    extern_metrics_list: List[Dict[str, Dict[str, List[float]]]] = None,
    )-> Tuple[Dict[str, Tuple[float, float]], np.ndarray]:
  """Compare a set of metrics using accuracy.

  This is a special case of CompareMetrics that uses pairwise accuracy
  (https://arxiv.org/abs/2107.10821) rather than standard correlation metrics.
  It assumes system-level granularity, and supports comparisons across multiple
  EvalSets (typically for different language pairs).

  Args:
    evs_list: List of EvalSets to use. The metrics to be compared will be
      limited to those whose basenames are available in all sets.
    main_refs_list: List of reference sets to be used in successive EvalSets.
    close_refs_list: List of reference sets to be excluded if include_human is
      True.
    include_human: Include any available human outputs, subject to the above
      constraints.
    include_outliers: Include any systems considered to be outliers.
    gold_name: Name of human gold scores to use, either a fixed value like 'mqm'
      that is available in all EvalSets, or 'std' to pick out the standard gold
      score in each EvalSet.
    primary_metrics: Include primary metrics only.
    domain: If not None, must be a value in evs.domain_names; this indicates
      that only the scores pertaining to that domain should be used.
    k: Number of resampling runs for PermutationSigDiff test. If k is 0, no
      test is performed, and only the raw ranks of metrics are returned, along
      with an empty significance matrix.
    psd: Minor params for PermutationSigDiff.
    pval: p-value for determining significant differences in ranks.
    extern_metrics_list: List of dicts containing external_metrics, one per
      evalset in evs_list. Each dict should map metric names to dicts containing
      scores for all systems, at system-level granularity (ie, scores are
      single-element lists).

  Returns:
    Tuple of (metric->(corr,rank), significance matrix). This is in the same
      format as CompareMetrics, except that metrics are DisplayName versions.
  """
  corrs, base_metrics = [], []
  if extern_metrics_list is None:
    extern_metrics_list = [None] * len(evs_list)
  for evs, main_refs, close_refs, extern_metrics in zip(
      evs_list, main_refs_list, close_refs_list, extern_metrics_list):
    corrs.append(GetCorrelations(
        evs, 'sys', main_refs, close_refs, include_human, include_outliers,
        gold_name, primary_metrics, domain, extern_metrics))
    base_metrics.append({evs.DisplayName(m): m for m in corrs[-1]})

  # Merge correlations across eval-sets, recording number of systems for each.
  num_sys_per_evs = [list(c.values())[0].num_sys for c in corrs]
  num_sys = sum(num_sys_per_evs)
  merged_corrs = {}
  for base_metric in set.intersection(*(set(b) for b in base_metrics)):
    gold_scores, metric_scores = [], []
    for c, metric_map in zip(corrs, base_metrics):
      metric = metric_map[base_metric]
      gold_scores.extend(c[metric].gold_scores)
      metric_scores.extend(c[metric].metric_scores)
    assert len(gold_scores) == num_sys, (len(gold_scores), num_sys)
    merged_corrs[base_metric] = stats.Correlation(
        num_sys, gold_scores, metric_scores)

  # Aggregated pairwise accuracy
  def _Accuracy(vect1, vect2):
    acc, num_pairs, b = 0, 0, 0
    for n in num_sys_per_evs:
      a, p = stats.Agreement(vect1[b: b + n], vect2[b: b + n])
      acc += a
      num_pairs += p
      b += n
    return acc / num_pairs, num_pairs

  # Compute metric correlations, ordered by decreasing correlation.
  corrs_and_ranks = {}
  for m, c in merged_corrs.items():
    corrs_and_ranks[m] = [_Accuracy(c.gold_scores, c.metric_scores)[0], 0]
  corrs_and_ranks = dict(
      sorted(corrs_and_ranks.items(), key=lambda x: -x[1][0]))

  # Compute significance matrix and determine ranks.
  sig_matrix = ComputeSigMatrix(
      merged_corrs, corrs_and_ranks, _Accuracy, 'none', k, psd, False)
  ranks = AssignRanks(sig_matrix, pval)
  for i, m in enumerate(corrs_and_ranks):
    corrs_and_ranks[m][1] = ranks[i]

  return corrs_and_ranks, sig_matrix


def ComputeSigMatrix(
    metric_corrs: Dict[str, stats.Correlation],
    corrs_and_ranks: Dict[str, tuple[float, int]],
    corr_fcn: Callable[[List[float], List[float], ...], Tuple[float, float]],
    average_by: str,
    k: int,
    psd: stats.PermutationSigDiffParams,
    replace_nans_with_zeros: bool,
    **corr_fcn_args,
) -> np.ndarray:
  """Populate significance matrix using PermutationSigDiff with given args."""
  n = len(corrs_and_ranks)
  sig_matrix = np.zeros((n, n))
  if not k:
    return sig_matrix
  for i in range(n):
    m1 = list(corrs_and_ranks)[i]
    for j in range(i + 1, n):
      m2 = list(corrs_and_ranks)[j]
      pval, _, _ = stats.PermutationSigDiff(
          metric_corrs[m2], metric_corrs[m1], corr_fcn, average_by, k,
          psd, replace_nans_with_zeros, **corr_fcn_args)
      sig_matrix[i, j] = pval
  return sig_matrix


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


def PrintMetricComparison(ranks, matrix, pval=0.05, evs=None, file=sys.stdout):
  """Pretty print the output from CompareMetrics*()."""
  if not evs:
    max_len = max(len(m) for m in ranks)
  else:
    max_len = max(len(evs.DisplayName(m)) for m in ranks)
  for i, metric in enumerate(ranks):
    s, r = ranks[metric]
    sig = ' '.join(['.'] * (i + 1)) + ' '
    sig += ' '.join('>' if p < pval else '=' for p in matrix[i][i + 1:])
    metric = evs.DisplayName(metric) if evs else metric
    print(f'{metric:<{max_len}} {r:>2} {s:6.3f}  {sig}', file=file)


# pylint: disable=unused-argument
def MakeTaskName(
    test_set, lang, domain, level, human, avg_by, corr_fcn, k, gold, refs,
    close_refs=None, use_outliers=False, primary=True, pval=0.05,
    block_size=1000, early_min=0.02, early_max=0.50,
    replace_nans_with_zeros=False, **corr_fcn_args) -> str:
  """Make a task name from a set of values assigned to attributes."""
  # Named parameters are just to standardize order.
  # pylint: disable=unused-variable
  corr_fcn_args = dict(sorted(corr_fcn_args.items()))
  vals = [f'{v}'.replace(' ', '') for v in locals().values()]
  return ' '.join(f'{k}={v}' for k, v in zip(locals(), vals))
