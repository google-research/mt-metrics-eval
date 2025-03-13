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

import ast
import collections
import copy
import functools
import itertools
import json
import os
import sys
import tarfile
from typing import Any, Callable, Iterable, Sequence
import urllib.request
import apache_beam as beam
from mt_metrics_eval import meta_info
from mt_metrics_eval import ratings
from mt_metrics_eval import stats
import numpy as np

import glob




TGZ = 'https://storage.googleapis.com/mt-metrics-eval/mt-metrics-eval-v2.tgz'

# This string is used to represent a missing or null translation in a text file.
# This is for standarization purposes only. If this string is on a line in a
# text file, it will be interpreted as `None` instead of the string itself.
NO_TRANSLATION = '[No translation supplied]'


class EvalSet:
  """Data for a test set and one language pair."""

  def __init__(self, name: str,
               lp: str,
               read_stored_metric_scores: bool = False,
               info: meta_info.MetaInfo = None,
               path: str = None,
               strict: bool = False,
               read_stored_ratings: bool = False):
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
      read_stored_ratings: Read stored ratings (character-level error spans) if
        any exist for this dataset. This makes loading slower, and is only
        needed for analysis that directly involves ratings.
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
    self._primary_metrics = self.info.primary_metrics.copy()

    self._ReadDataset(
        name, lp, read_stored_metric_scores, path, strict, read_stored_ratings)

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
  def src_lang(self) -> str:
    return self.lp.split('-')[0]

  @property
  def tgt_lang(self) -> str:
    return self.lp.split('-')[1]

  @property
  def levels(self) -> set[str]:
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
  def ref_names(self) -> set[str]:
    """Names of available references."""
    return set(self._all_refs.keys())

  @property
  def std_ref(self) -> str:
    """Name of standard reference."""
    return self.info.std_ref

  @property
  def sys_names(self) -> set[str]:
    """Names of all 'systems' for which output is available."""
    return set(self._sys_outputs.keys())

  @property
  def human_sys_names(self) -> set[str]:
    """Names of systems in sys_names that are human output."""
    return self._human_sys_names

  @property
  def outlier_sys_names(self) -> set[str]:
    """Names of systems in sys_names considered to be outliers."""
    return self.info.outlier_systems

  def SetOutlierSysNames(self, outliers: set[str]) -> None:
    """Overwrites the list of outlier systems."""
    self.info.outlier_systems = outliers

  @property
  def human_score_names(self) -> set[str]:
    """Names of different human scores available, eg 'wmt-z', 'mqm'."""
    return self._human_score_names

  @property
  def human_rating_names(self) -> set[str]:
    """Names of different human ratings available, eg 'mqm.rater1'."""
    return self._human_rating_names

  def StdHumanScoreName(self, level) -> str:
    """Name of standard human score for a given level in self.levels."""
    if level in self._std_human_scores:
      return self._std_human_scores[level]
    else:
      return None

  @property
  def metric_names(self) -> set[str]:
    """Full names of available metrics, eg BLEU-refA, COMET-refB."""
    return self._metric_names

  @property
  def metric_basenames(self) -> set[str]:
    """Basenames of available metrics, eg BLEU, COMET."""
    return self._metric_basenames

  @property
  def primary_metrics(self) -> set[str]:
    """Base names of primary metrics, empty set if none."""
    return self._primary_metrics

  def BaseMetric(self, metric_name: str) -> str:
    """Base name for a given metric, eg BLEU for BLEU-refA."""
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
    elif fmt == 'full':
      name = metric_name
    elif fmt == 'base':
      pass
    else:
      raise ValueError(f'Unkown format: {fmt}')
    return name

  def ReferencesUsed(self, metric_name: str) -> set[str]:
    """Reference(s) used by a metric."""
    return self.ParseMetricName(metric_name)[1]

  @property
  def domains(self) -> dict[str, list[list[int]]]:
    """Map from domain name to [[beg, end+1], ...] segment position lists."""
    return self._domains

  @property
  def docs(self) -> dict[str, list[int]]:
    """Map from doc name to [beg, end+1] segment positions."""
    return self._docs

  @property
  def src(self) -> list[str]:
    """Segments in the source text, in order."""
    return self._src

  @property
  def all_refs(self) -> dict[str, list[str]]:
    """Map from reference name to text for that reference."""
    return self._all_refs

  @property
  def sys_outputs(self) -> dict[str, list[str]]:
    """Map from system name to output text from that system."""
    return self._sys_outputs

  def Scores(self, level: str, scorer: str) -> dict[str, list[float]]:
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

  @property
  def rating_names(self) -> set[str]:
    """The names of all available sets of ratings."""
    return set(self._ratings.keys())

  def Ratings(self, rating_name) -> dict[str, list[ratings.Rating | None]]:
    """Get stored ratings assigned to segments.

    Args:
      rating_name: Any string in human_rating_names.

    Returns:
      Mapping from system names to lists of Ratings objects corresponding to
      segments in order. Each Rating is the output of a single rater working on
      the corresponding segment. (Some rater names include 'merged' to indicate
      that different raters have worked on different segments, but these still
      have only one rater per segment.) None entries mean the segment hasn't
      been rated; empty 'errors' members in the Ratings objects mean it is
      judged to be error free.
    """
    return self._ratings[rating_name] if rating_name in self._ratings else None

  def RaterIdsPerSeg(self, rating_name: str) -> dict[str, list[str | None]]:
    """Returns a dict of system to rater IDs for each segment."""
    return self._rater_ids[rating_name]

  @property
  def metadata(self) -> dict[str, list[dict[str, Any]]]:
    """A dict of system to extra metadata for each translation."""
    return self._metadata

  def Correlation(self,
                  gold_scores: dict[str, list[float]],
                  metric_scores: dict[str, list[float]],
                  sys_names: Iterable[str] = None,
                  indexes: Sequence[int] = None):
    """Get correlation statistics for given metric scores.

    Args:
      gold_scores: Gold scores to use, same format as metric_scores, except that
        score lists may contain None values.
      metric_scores: Metric scores to evaluate, a map from system names to lists
        of float scores.
      sys_names: Names of systems to use in comparison, must exist in both
        metric_scores and gold_scores. Default is to use all systems for which
        gold scores are available.
      indexes: Optional sequence of indexes to select from each scores list.

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
      if indexes is not None:
        gscores = np.asarray(gscores)[indexes]
        mscores = np.asarray(mscores)[indexes]
      if len(gscores) != len(mscores):
        raise ValueError('Wrong number of scores for system %s: %d vs %d' %
                         (sys_name, len(gscores), len(mscores)))
      all_gold_scores.extend(gscores)
      all_metric_scores.extend(mscores)
    return stats.Correlation(len(sys_names), all_gold_scores, all_metric_scores)

  def ParseHumanScoreFilename(self, filename, rating_file=False):
    """Parse a human-score/rating filename into lang, name, level components."""
    exten = 'rating' if rating_file else 'score'
    # SRC-TGT.NAME.LEVEL.score
    toks = os.path.basename(filename).split('.')
    if len(toks) < 4 or toks[-1] != exten:
      raise ValueError(f'Bad format for human {exten}s: {filename}')
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

  def SetPrimaryMetrics(self, metrics: set[str]):
    """Set primary metrics to the given set of basenames. Reset if empty."""
    if metrics:
      self._primary_metrics = metrics.copy()
    else:
      self._primary_metrics = self.info.primary_metrics.copy()

  def AddMetricsFromDir(self, dir_name, repair=False, replace=False):
    """Add metrics from files in a directory.

    This can be used to add new metrics post-construction. It is not affected
    by the read_stored_metric_scores flag, so you can construct an EvalSet with
    no metrics, then add metrics from an arbitrary directory using this
    function. Any files that end in .score will be interpreted as metrics, and
    must adhere to the conventions described in the README.

    Args:
      dir_name: Name of directory containing files of the form
        NAME-REF.LEVEL.score whose contents are to be added. Metric names must
        not overlap with existing metrics unless replace is True.
      repair: Replace missing scores with 0s, see CheckScores().
      replace: If True, overwrite scores for any existing metric with the same
        name.

    Returns:
      List of basenames for metrics read in.
    """
    basenames = []
    for filename in glob.glob(os.path.join(dir_name, '*.score')):
      name, level = self.ParseMetricFilename(filename)
      basename, refs = self.ParseMetricName(name)
      if level == 'domain':
        scores = ReadDomainScoreFile(filename, self.domain_names)
      else:
        scores = ReadScoreFile(filename)
      self.AddMetric(basename, refs, level, scores, repair, replace)
      basenames.append(basename)
    return basenames

  def AddMetric(
      self,
      basename: str,
      refs: set[str],
      level: str,
      scores: dict[str, list[float]],
      repair: bool = False,
      replace: bool = False):
    """Add a new metric to the EvalSet.

    Args:
      basename: Basename of metric to add (eg BLEU, not BLEU-refA). Must not
        already exist at this level unless replace is True.
      refs: Reference(s) used by the metric, a subset of self.ref_names, or the
        empty set to indicate a source-based metric.
      level: Granularity, one of 'sys', 'domain', 'doc', or 'seg'.
      scores: Mapping from each system name to correctly ordered list of scores,
        a singleton list if level is 'sys', otherwise the order in which items
        occur in self.domains, self.docs, or self.src.
      repair: Replace missing scores with 0s, see CheckScores().
      replace: If True, overwrite scores for any existing metric with the same
        name.
    """
    if not refs.issubset(self.ref_names):
      raise ValueError(f'Bad reference(s) for metric {basename}: {refs}')
    if refs == self.ref_names and len(refs) >= 3:
      refs = 'all'
    name = MakeMetricName(basename, refs)

    if level not in self._scores:
      self._scores[level] = {}
    if name in self._scores[level]:
      if not replace:
        raise ValueError(f'Duplicate metric name {name} at {level} level')
    else:
      self._metric_names.add(name)
      self._metric_basenames.add(basename)
    self._scores[level][name] = scores
    self.CheckScores(self._scores[level][name], name, level, False, repair)

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
    """Return a list containing the document name for each segment, in order."""
    return _UnmapPositions(self.docs, contiguous=True)

  def DomainsPerSeg(self):
    """Return a list containng the domain name for each segment, in order."""
    return _UnmapPositions(self.domains, contiguous=False)

  def DomainsPerDoc(self):
    """Return a list containing the domain name for each document, in order."""
    domains_and_docs = zip(self.DomainsPerSeg(), self.DocsPerSeg())
    return [domain for domain, _ in _MapPositions(list(domains_and_docs))]

  def _ReadDataset(
      self, name, lp, read_stored_metric_scores, path, strict,
      read_stored_ratings):
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
    self._src = _ReadSourceFile(os.path.join(d, 'sources'), lp)

    self._all_refs = {}
    for filename in glob.glob(os.path.join(d, 'references', '%s.*.*' % lp)):
      refname = filename.split('.')[-2]
      if '-' in refname or refname in ['all', 'src']:
        assert False, f'Invalid reference name: {refname}'
      if refname in self._all_refs:
        raise ValueError(f'Duplicate reference name: {refname}')
      self._all_refs[refname] = _ReadReferenceFile(filename)

    self._outlier_sys_names, self._human_sys_names = set(), set()
    self._sys_outputs = {}
    for filename in glob.glob(os.path.join(d, 'system-outputs', lp, '*.*')):
      extension = os.path.splitext(filename)[1]
      sysname = os.path.basename(filename)[:-len(extension)]
      self._sys_outputs[sysname] = _ReadSystemOutputFile(filename)
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
      self.CheckScores(
          self._scores[level][scorer], scorer, level, True, repair=not strict)

    self._human_rating_names = set()
    self._ratings = {}
    self._rater_ids = {}
    if read_stored_ratings:
      for filename in glob.glob(
          os.path.join(d, 'human-scores', f'{lp}.*.rating')
      ):
        _, rating_name, level = self.ParseHumanScoreFilename(
            os.path.basename(filename), rating_file=True
        )
        assert level == 'seg'
        self._human_rating_names.add(rating_name)
        assert rating_name not in self._ratings, rating_name
        self._ratings[rating_name], self._rater_ids[rating_name] = (
            ratings.ReadRatingFile(filename, rating_name)
        )

    self._metric_names = set()
    self._metric_basenames = set()
    if read_stored_metric_scores:
      for md in metric_scores_paths:
        md = os.path.join(md, name, 'metric-scores', lp)
        self.AddMetricsFromDir(md, repair=not strict)

    # Load metadata for the translations.
    self._metadata = {}
    for filename in glob.glob(os.path.join(d, 'metadata', lp, '*.jsonl')):
      extension = os.path.splitext(filename)[1]
      sysname = os.path.basename(filename)[:-len(extension)]
      self._metadata[sysname] = _ReadMetadataFile(filename)

    # Check contents
    for txt in self.all_refs.values():
      assert len(txt) == len(self.src), f'Bad length for reference {txt}'
    for txt in self.sys_outputs.values():
      assert len(txt) == len(self.src), f'Bad length for output {txt}'


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
    lines = []
    for line in f:
      line = line.rstrip()
      if line == NO_TRANSLATION:
        lines.append(None)
      else:
        lines.append(line)
  return lines


def _ReadFieldFromJsonl(filename: str, field: str) -> list[str]:
  values = []
  with open(filename) as f:
    for line in f:
      example = json.loads(line)
      values.append(example[field])
  return values


def _ReadSourceFile(source_dir: str, lp: str) -> list[str]:
  txt_file = os.path.join(source_dir, '%s.txt' % lp)
  jsonl_file = os.path.join(source_dir, '%s.jsonl' % lp)
  if os.path.exists(txt_file):
    return _ReadTextFile(txt_file)
  elif os.path.exists(jsonl_file):
    return _ReadFieldFromJsonl(jsonl_file, 'source')
  else:
    raise ValueError(f'No source file found for {lp}')


def _ReadReferenceFile(filename: str) -> list[str]:
  if filename.endswith('.txt'):
    return _ReadTextFile(filename)
  elif filename.endswith('.jsonl'):
    return _ReadFieldFromJsonl(filename, 'target')
  else:
    raise ValueError(f'Unsupported reference file type: {filename}')


def _ReadSystemOutputFile(filename: str) -> list[str]:
  if filename.endswith('.txt'):
    return _ReadTextFile(filename)
  elif filename.endswith('.jsonl'):
    return _ReadFieldFromJsonl(filename, 'hypothesis')
  else:
    raise ValueError(f'Unsupported system output file type: {filename}')


def _ReadMetadataFile(filename: str) -> list[dict[str, Any]]:
  metadata = []
  with open(filename, 'r') as f:
    for line in f:
      metadata.append(json.loads(line))
  return metadata


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


def GetCorrelations(evs: EvalSet, level: str, main_refs: set[str],
                    close_refs: set[str], include_human: bool,
                    include_outliers: bool, gold_name: str,
                    primary_metrics: bool, domain: str = None,
                    extern_metrics: dict[str, dict[str, list[float]]] = None,
                    sample_size: int = None, sample_method: str = None,
                    sample_seed: int = None,
                    metric_format: str = 'full',
                    ) -> dict[str, stats.Correlation]:
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
    sample_size: If not None, a number of items (segments) to sample. This
      is a no-op if sample_size is 0 or greater than the number of  items
      available. Only applies to segment-level granularity.
    sample_method: Any of the methods in stats.Sample. If 'stratify', does
      stratified sampling over documents (filtered by domain if domain is not
      None).
    sample_seed: Random seed for sampling, None for a fresh draw.
    metric_format: Format to use for metric names, see DisplayName().

  Returns:
    Map from metric names to stats.Correlation objects from which correlation
    and significance statistics can be computed.
  """
  if domain is not None:
    assert domain in evs.domain_names
    if level == 'sys':
      level = 'domain'
  assert level in evs.levels

  sample = None
  if sample_size is not None:
    doc_sizes = [e - b for b, e in evs.docs.values()]
    if domain is not None:
      doc_sizes = [
          s for d, s in zip(evs.DomainsPerDoc(), doc_sizes)  if d == domain]
    total_size = sum(doc_sizes)
    sample = stats.Sample(
        total_size, sample_size, sample_method, doc_sizes, sample_seed)

  def _Filter(scores):
    if domain is not None:
      mask = {
          'domain': [d == domain for d in evs.domain_names],
          'doc': [d == domain for d in evs.DomainsPerDoc()],
          'seg': [d == domain for d in evs.DomainsPerSeg()]
          }[level]
      filt_scores = {}
      for k, vals in scores.items():
        assert len(vals) == len(mask)
        filt_scores[k] = [s for s, m in zip(vals, mask) if m]
      scores = filt_scores
    if sample is not None:
      scores = {k: sample.Select(s) for k, s in scores.items()}
    return scores

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

  # gold_scores may contain systems that don't have any gold scores. Select
  # just the subset of systems that does.
  gold_scores = {
      system: scores for system, scores in gold_scores.items()
      if scores and any(score is not None for score in scores)
  }

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
    display_name = evs.DisplayName(metric_name, metric_format)
    correlations[display_name] = evs.Correlation(
        gold_scores, _Filter(metric_scores), sys_names)

    # Add in extra metrics.
    if extern_metrics is not None:
      for metric_name, scores in extern_metrics.items():
        correlations[metric_name] = evs.Correlation(
            gold_scores, _Filter(scores), sys_names)

  return correlations


def CompareMetrics(
    metric_corrs: dict[str, stats.Correlation],
    corr_fcn: Callable[[list[float], list[float], ...], tuple[float, float]],
    average_by: str = 'none',
    k: int = 1000,
    psd: stats.PermutationSigDiffParams = stats.PermutationSigDiffParams(),
    pval: float = 0.05,
    replace_nans_with_zeros: bool = False,
    perm_test: str = 'scores',
    parallel_file: str = None,
    **corr_fcn_args,
    ) -> tuple[
        dict[str, tuple[float, float]], np.ndarray, np.ndarray, np.ndarray]:
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
    perm_test: Permutation test to use, either 'scores' or 'pairs' to select
      stats.PermutationSigDiff or stats.PairwisePermutationSigDiff. In the
      latter case, corr_fcn is ignored, and the 'variant' and 'sample_rate'
      flags from corr_fcn_args are interpreted as arguments to
      KendallWithTiesOpt.
    parallel_file: If not None, the significance matrix will be computed in
      parallel using beam, with this value as the name of a temporary file for
      beam output.

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
    3. Draws index: a square numpy array with the same dimensions as sig_matrix,
       draws_index[i, j] and draws_index[j, i] give the start and end+1 indexes
       in draws_list for metric pair i, j, j > i.
    4. Draws list: an K x 2 numpy array of resampling results. draws_list[k]
       contains a pair of resampled correlations for metrics i, j as indicated
       by draws_index.
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
  # Use metric name as secondary sort criterion to stablize ties.
  corrs_and_ranks = dict(
      sorted(corrs_and_ranks.items(), key=lambda x: (-x[1][0], x[0])))

  # Compute significance matrix and determine ranks.
  sig_matrix, draws_index, draws_list = ComputeSigMatrix(
      metric_corrs, corrs_and_ranks, corr_fcn, average_by, k,
      psd, replace_nans_with_zeros, perm_test, parallel_file, **corr_fcn_args)
  ranks = AssignRanks(sig_matrix, pval)
  for i, m in enumerate(corrs_and_ranks):
    corrs_and_ranks[m][1] = ranks[i]

  return corrs_and_ranks, sig_matrix, draws_index, draws_list


def CompareMetricsWithGlobalAccuracy(
    evs_list: list[EvalSet],
    main_refs_list: list[set[str]],
    close_refs_list: list[set[str]],
    include_human: bool,
    include_outliers: bool,
    gold_name: str | list[str],
    primary_metrics: bool,
    domain: str = None,
    k: int = 1000,
    psd: stats.PermutationSigDiffParams = stats.PermutationSigDiffParams(),
    pval: float = 0.05,
    extern_metrics_list: list[dict[str, dict[str, list[float]]]] = None,
    parallel_file: str = None,
    )-> tuple[
        dict[str, tuple[float, float]], np.ndarray, np.ndarray, np.ndarray]:
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
    gold_name: Name(s) of human gold scores to use. If this is a single string,
      it must be a value like 'mqm' that is available in all EvalSets, or 'std'
      to pick out the standard gold score in each EvalSet. If it is a list, it
      must contain the scorer to use for each element in eval_list.
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
    parallel_file: If not None, the significance matrix will be computed in
      parallel using beam, with this value as the name of a temporary file for
      beam output.

  Returns:
    Same tuple as CompareMetrics. Metric names in the metric->(corr, rank) map
    are DisplayName versions.
  """
  corrs, base_metrics = [], []
  if extern_metrics_list is None:
    extern_metrics_list = [None] * len(evs_list)
  if isinstance(gold_name, str):
    gold_name = [gold_name] * len(evs_list)
  for evs, main_refs, close_refs, gold, extern_metrics in zip(
      evs_list, main_refs_list, close_refs_list, gold_name,
      extern_metrics_list):
    corrs.append(GetCorrelations(
        evs, 'sys', main_refs, close_refs, include_human, include_outliers,
        gold, primary_metrics, domain, extern_metrics))
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
  # Use metric name as secondary sort criterion to stablize ties.
  corrs_and_ranks = dict(
      sorted(corrs_and_ranks.items(), key=lambda x: (-x[1][0], x[0])))

  # Compute significance matrix and determine ranks.
  sig_matrix, draws_index, draws_list = ComputeSigMatrix(
      merged_corrs, corrs_and_ranks, _Accuracy, 'none', k, psd, False, 'scores',
      parallel_file)
  ranks = AssignRanks(sig_matrix, pval)
  for i, m in enumerate(corrs_and_ranks):
    corrs_and_ranks[m][1] = ranks[i]

  return corrs_and_ranks, sig_matrix, draws_index, draws_list


def CompareMetricsWithPairwiseConfidenceError(
    metric_corrs: dict[str, stats.Correlation],
    k: int = 1000,
    psd: stats.PermutationSigDiffParams = stats.PermutationSigDiffParams(),
    pval: float = 0.05,
    replace_nans_with_zeros: bool = False,
    perm_test: str = 'scores',
    parallel_file: str = None,
) -> tuple[dict[str, tuple[float, float]], np.ndarray, np.ndarray, np.ndarray]:
  """Compare a set of metrics using pairwise confidence error."""
  assert metric_corrs

  first_corr = list(metric_corrs.values())[0]
  pce_wrapper = functools.partial(
      stats.PairwiseConfidenceError,
      num_sys=first_corr.num_sys,
      filter_nones=first_corr.none_count > 0,
  )

  # Compute metric correlations, ordered by decreasing correlation.
  corrs_and_ranks = {}
  for m, c in metric_corrs.items():
    corrs_and_ranks[m] = [pce_wrapper(c.gold_scores, c.metric_scores)[0], 0]
  # Use metric name as secondary sort criterion to stablize ties.
  corrs_and_ranks = dict(
      sorted(corrs_and_ranks.items(), key=lambda x: (-x[1][0], x[0])))

  # Compute significance matrix and determine ranks.
  sig_matrix, draws_index, draws_list = ComputeSigMatrix(
      metric_corrs, corrs_and_ranks, pce_wrapper, 'none', k,
      psd, replace_nans_with_zeros, perm_test, parallel_file)
  ranks = AssignRanks(sig_matrix, pval)
  for i, m in enumerate(corrs_and_ranks):
    corrs_and_ranks[m][1] = ranks[i]

  return corrs_and_ranks, sig_matrix, draws_index, draws_list


def ComputeSigMatrix(
    metric_corrs: dict[str, stats.Correlation],
    corrs_and_ranks: dict[str, tuple[float, int]],
    corr_fcn: Callable[[list[float], list[float], ...], tuple[float, float]],
    average_by: str,
    k: int,
    psd: stats.PermutationSigDiffParams,
    replace_nans_with_zeros: bool,
    perm_test: str,
    parallel_file: str = None,
    **corr_fcn_args,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Populate significance matrix using PermutationSigDiff with given args."""

  variant, sample_rate = 'acc23', 1.0
  if perm_test == 'pairs':
    if 'variant' in corr_fcn_args:
      variant = corr_fcn_args['variant']
    if 'sample_rate' in corr_fcn_args:
      sample_rate = corr_fcn_args['sample_rate']
  elif perm_test != 'scores':
    raise ValueError(f'Bad perm_test value: {perm_test}')

  n, metrics = len(corrs_and_ranks), list(corrs_and_ranks)
  sig_matrix, draws_index = np.zeros((n, n)), np.zeros((n, n), dtype=np.int32)
  draws_list = []  # List of paired correlations from resampling draws
  if not k:
    return sig_matrix, draws_index, np.asarray(draws_list)

  def ComputePval(
      metric1, metric2
  ) -> tuple[str, str, float, list[tuple[float, float]]]:
    if perm_test == 'scores':
      pval, _, _, draws = stats.PermutationSigDiff(
          metric_corrs[metric2], metric_corrs[metric1], corr_fcn, average_by, k,
          psd, replace_nans_with_zeros, **corr_fcn_args)
    elif perm_test == 'pairs':
      pval, _, _, draws = stats.PairwisePermutationSigDiff(
          metric_corrs[metric2], metric_corrs[metric1], variant, average_by, k,
          psd, sample_rate=sample_rate,
          replace_nans_with_zeros=replace_nans_with_zeros)
    else:
      pval, draws = 0, None
      assert False
    return metric1, metric2, pval, draws

  def Collate(i, j, pval, draws):
    """Collate results from i, j comparison."""
    sig_matrix[i, j] = pval
    draws_index[i, j] = len(draws_list)
    draws_index[j, i] = len(draws_list) + len(draws)
    draws_list.extend(draws)

  if parallel_file is None:
    for i in range(n):
      for j in range(i + 1, n):
        _, _, pval, draws = ComputePval(metrics[i], metrics[j])
        Collate(i, j, pval, draws)
  else:
    todo = []
    for i in range(n):
      for j in range(i + 1, n):
        todo.append((metrics[i], metrics[j]))
    
    with beam.Pipeline() as p:
      _ = (p
           | beam.Create(todo)
           | beam.MapTuple(ComputePval)
           | beam.CombineGlobally(beam.combiners.ToListCombineFn())
           | beam.io.WriteToText(parallel_file, shard_name_template=''))
    with open(parallel_file, 'r') as f:
      line = f.read().strip()
    for m1, m2, p, draws in ast.literal_eval(line):
      Collate(metrics.index(m1), metrics.index(m2), p, draws)
    os.remove(parallel_file)

  return sig_matrix, draws_index, np.asarray(draws_list)


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


def PrintMetricComparison(
    ranks, matrix, pval=0.05, evs=None, file=sys.stdout, probs=False):
  """Pretty print the output from CompareMetrics*()."""
  if not evs:
    max_len = max(len(m) for m in ranks)
  else:
    max_len = max(len(evs.DisplayName(m)) for m in ranks)
  for i, metric in enumerate(ranks):
    s, r = ranks[metric]
    if probs:
      sig = ' '.join(['  .  '] * (i + 1)) + ' '
      sig += ' '.join(f'{p:5.3f}' for p in matrix[i][i + 1:])
    else:
      sig = ' '.join(['.'] * (i + 1)) + ' '
      sig += ' '.join('>' if p < pval else '=' for p in matrix[i][i + 1:])
    metric = evs.DisplayName(metric) if evs else metric
    print(f'{metric:<{max_len}} {r:>2} {s:10.7f}  {sig}', file=file)


def MakeMetricName(basename, refs):
  """Make metric full name from basename and references used (not checked)."""
  if isinstance(refs, str):
    refs = {refs}
  if refs == {'all'}:
    ref_string = 'all'  # reserved for using all references
  elif refs:
    ref_string = '.'.join(refs)
  else:
    ref_string = 'src'
  return f'{basename}-{ref_string}'


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
