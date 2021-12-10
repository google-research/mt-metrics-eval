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
import os
import tarfile
from typing import Dict, Iterable, List, Sequence, Set
import urllib.request
from mt_metrics_eval import stats
import glob



TGZ = 'https://storage.googleapis.com/mt-metrics-eval/mt-metrics-eval.tgz'

DATA = {
    'wmt21.news': {
        'language_pairs': [
            'en-de', 'en-ru', 'zh-en',
            'cs-en', 'de-en', 'ha-en', 'is-en', 'ja-en', 'ru-en',
            'de-fr', 'en-cs', 'en-ha', 'en-is', 'en-ja', 'en-zh', 'fr-de',
        ],
        'std_scorers': {
            'sys': 'mqm',
            'seg': 'mqm',
        },
        'backup_scorer': 'wmt-z',
        'outlier_systems': {},
    },
    'wmt21.tedtalks': {
        'language_pairs': ['en-de', 'en-ru', 'zh-en'],
        'std_scorers': {
            'sys': 'mqm',
            'seg': 'mqm',
        },
        'backup_scorer': 'wmt-z',
        'outlier_systems': {},
    },
    'wmt21.flores': {
        'language_pairs': ['bn-hi', 'hi-bn', 'xh-zu', 'zu-xh'],
        'std_scorers': {
            'sys': 'wmt-z',
            'seg': 'wmt-raw',
        },
        'outlier_systems': {},
    },
    'wmt20': {
        'language_pairs': [
            'cs-en', 'de-en', 'en-cs', 'en-de', 'en-iu', 'en-ja', 'en-pl',
            'en-ru', 'en-ta', 'en-zh', 'iu-en', 'ja-en', 'km-en', 'pl-en',
            'ps-en', 'ru-en', 'ta-en', 'zh-en'
        ],
        'std_scorers': {
            'sys': 'wmt-z',
            'doc': 'wmt-raw',
            'seg': 'wmt-raw'
        },
        'outlier_systems': {
            'cs-en': ['zlabs-nlp.1149', 'CUNI-DocTransformer.1457'],
            'de-en': ['yolo.1052', 'zlabs-nlp.1153', 'WMTBiomedBaseline.387'],
            'iu-en': ['NiuTrans.1206', 'Facebook_AI.729'],
            'ja-en': ['Online-G.1564', 'zlabs-nlp.66', 'Online-Z.1640'],
            'pl-en': ['zlabs-nlp.1162'],
            'ru-en': ['zlabs-nlp.1164'],
            'ta-en': ['Online-G.1568', 'TALP_UPC.192'],
            'zh-en': ['WMTBiomedBaseline.183'],
            'en-cs': ['zlabs-nlp.1151', 'Online-G.1555'],
            'en-de': [
                'zlabs-nlp.179', 'WMTBiomedBaseline.388', 'Online-G.1556'
            ],
            'en-iu': ['UEDIN.1281', 'OPPO.722', 'UQAM_TanLe.521'],
            'en-pl': ['Online-Z.1634', 'zlabs-nlp.180', 'Online-A.1576'],
            'en-ta': ['TALP_UPC.1049', 'SJTU-NICT.386', 'Online-G.1561'],
            'en-ja': ['Online-G.1557', 'SJTU-NICT.370']
        }
    },
    'wmt19': {
        'language_pairs': [
            'de-cs', 'de-en', 'de-fr', 'en-cs', 'en-de', 'en-fi', 'en-gu',
            'en-kk', 'en-lt', 'en-ru', 'en-zh', 'fi-en', 'fr-de', 'gu-en',
            'kk-en', 'lt-en', 'ru-en', 'zh-en'
        ],
        'std_scorers': {
            'sys': 'wmt-z',
            'seg': 'wmt-raw'
        },
        'outlier_systems': {}
    }
}


class EvalSet:
  """Data for an evaluation set and one language pair."""

  def __init__(self, name, lp, read_stored_metric_scores=False):
    """Load dataset for a given language pair, eg EvalSet('wmt20', 'en-de').

    Args:
      name: Name of dataset, any top-level key in DATA.
      lp: Language pair, any key in DATA[name]['language_pair'].
      read_stored_metric_scores: Read stored scores for automatic metrics for
        this dataset. This makes loading slower, and is only needed for
        analyzing or directly comparing to these scores.
    """
    if name not in DATA:
      raise ValueError('Unknown dataset: %s' % name)
    if lp not in DATA[name]['language_pairs']:
      raise ValueError('Language pair not in %s: %s' % (name, lp))

    self._ReadDataset(name, lp, read_stored_metric_scores)

  @property
  def doc_names(self) -> Sequence[str]:
    """Names of documents, in order."""
    return self._docs.keys()

  @property
  def ref_names(self) -> Sequence[str]:
    """Names of available references."""
    return self._all_refs.keys()

  @property
  def sys_names(self) -> Sequence[str]:
    """Names of all 'systems' for which output is available."""
    return self._sys_outputs.keys()

  @property
  def human_sys_names(self) -> Set[str]:
    """Names of systems in sys_names that are human output."""
    return self._human_sys_names

  @property
  def outlier_sys_names(self) -> Set[str]:
    """Names of systems in sys_names considered to be outliers."""
    return self._outlier_sys_names

  @property
  def human_score_names(self) -> Set[str]:
    """Names of different human scores available."""
    return self._human_score_names

  @property
  def metric_names(self) -> Set[str]:
    """Names of different metric scores available."""
    return self._metric_names

  @property
  def docs(self) -> Dict[str, List[int]]:
    """Map from doc name to [beg, end+1] segment positions."""
    return self._docs

  @property
  def src(self) -> List[str]:
    """Segments in the source text, in order."""
    return self._src

  @property
  def ref(self) -> List[str]:
    """Segments in the standard reference text, in order."""
    return self._all_refs['ref']

  @property
  def all_refs(self) -> Dict[str, List[str]]:
    """Map from reference name to text for that reference."""
    return self._all_refs

  @property
  def sys_outputs(self) -> Dict[str, List[str]]:
    """Map from system name to output text from that system."""
    return self._sys_outputs

  def Scores(self, level: str, scorer: str = 'std') -> Dict[str, List[float]]:
    """Get stored scores assigned to text units at a given level.

    Args:
      level: Text units to which scores apply, one of 'sys', 'doc', 'seg'.
      scorer: Method used to produce scores, may be any string in
        human_score_names or metric_names, or the special keyword 'std' which
        designates official human gold scores for the current level.

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
                  level: str,
                  metric_scores: Dict[str, List[float]],
                  gold_scorer: str = 'std',
                  sys_names: Iterable[str] = None):
    """Get correlation statistics for given metric scores.

    Args:
      level: Text units being scored, one of 'sys', 'doc', 'seg'.
      metric_scores: Metric scores to evaluate, a map from system names to lists
        of float scores. If level is 'sys', the lists must contain one score,
        otherwise scores corresponding to documents or segments in order.
      gold_scorer: Gold scores to use, may be any string in human_score_names or
        metric_names, or the special keyword 'std' which designates official
        human gold scores for the current level.
      sys_names: Names of systems to use in comparison, must exist in both
        metric_scores and in the scores designated by gold_scorer. The default
        is to use all systems for which gold scores exist, except those in
        outlier_sys_names and human_sys_names.

    Returns:
      A Correlation object for computing correlation statistics.
    """
    if gold_scorer not in self._scores[level]:
      raise ValueError('No scores for %s at %s level.' % (gold_scorer, level))
    gold_scores = self._scores[level][gold_scorer]
    if sys_names is None:
      sys_names = set(gold_scores).difference(self.outlier_sys_names,
                                              self.human_sys_names)
    all_gold_scores, all_metric_scores = [], []
    for sys_name in sys_names:
      if sys_name not in gold_scores:
        raise ValueError('No scores for system %s in gold scorer' % sys_name)
      if sys_name not in metric_scores:
        raise ValueError('No scores for system %s in metric_scores' % sys_name)
      gscores, mscores = gold_scores[sys_name], metric_scores[sys_name]
      if len(gscores) != len(mscores):
        raise ValueError('Wrong number of scores for system %s: %d vs %d' %
                         (sys_name, len(gscores), len(mscores)))
      all_gold_scores.extend(gscores)
      all_metric_scores.extend(mscores)
    return stats.Correlation(len(sys_names), all_gold_scores, all_metric_scores)

  def _ReadDataset(self, name, lp, read_stored_metric_scores):
    """Read data for given name and language pair."""

    path = LocalDir(root_only=False)
    if not os.path.exists(path):
      raise ValueError('%s not found. Run mtme --download.' % path)

    d = os.path.join(path, name)
    self._name = name
    self._lp = lp
    self._docs = ReadDocPositions(os.path.join(d, 'documents', '%s.docs' % lp))
    self._src = ReadTextFile(os.path.join(d, 'sources', '%s.txt' % lp))
    self._all_refs = {}
    for filename in glob.glob(os.path.join(d, 'references', '%s.*.txt' % lp)):
      refname = filename.split('.')[-2]
      self._all_refs[refname] = ReadTextFile(filename)
    self._outlier_sys_names, self._human_sys_names = set(), set()
    self._sys_outputs = {}
    for filename in glob.glob(os.path.join(d, 'system-outputs', lp, '*.txt')):
      sysname = os.path.basename(filename)[:-len('.txt')]
      self._sys_outputs[sysname] = ReadTextFile(filename)
      if (lp in DATA[name]['outlier_systems'] and
          sysname in DATA[name]['outlier_systems'][lp]):
        self._outlier_sys_names.add(sysname)
      elif sysname.startswith('Human') or sysname.startswith('ref-'):
        self._human_sys_names.add(sysname)

    self._human_score_names = set()
    self._scores = {'sys': {}, 'doc': {}, 'seg': {}}
    for filename in glob.glob(
        os.path.join(d, 'human-scores', '%s.*.score' % lp)):
      _, scorer, level, _ = os.path.basename(filename).split('.')
      self._human_score_names.add(scorer)
      self._scores[level][scorer] = ReadScoreFile(filename)
    for level, sname in DATA[name]['std_scorers'].items():
      if sname in self._scores[level]:
        self._scores[level]['std'] = self._scores[level][sname]
      elif 'backup_scorer' in DATA[name]:
        sname = DATA[name]['backup_scorer']
        assert sname in self._scores[level], (sname, level)
        self._scores[level]['std'] = self._scores[level][sname]
      else:
        assert sname in self._scores[level], (sname, level)  # fail

    self._metric_names = set()
    if read_stored_metric_scores:
      for filename in glob.glob(
          os.path.join(d, 'metric-scores', lp, '*.score')):
        scorer, level, _ = os.path.basename(filename).rsplit('.', maxsplit=2)
        assert scorer not in self._scores[level]
        self._metric_names.add(scorer)
        self._scores[level][scorer] = ReadScoreFile(filename)

    for txt in self.all_refs.values():
      assert len(txt) == len(self.src)
    for txt in self.sys_outputs.values():
      assert len(txt) == len(self.src)
    expected_len = {'sys': 1, 'doc': len(self.docs), 'seg': len(self.src)}
    for level in 'sys', 'doc', 'seg':
      if level in self._scores:
        for scorer in self._scores[level].values():
          for name, scores in scorer.items():
            assert name in self.sys_names, (level, name)
            assert len(scores) == expected_len[level]


def LocalDir(root_only=True):
  """Location for local dir: $HOME/.mt-metrics-eval."""
  path = os.path.join(os.path.expanduser('~'), '.mt-metrics-eval')
  if not root_only:
    path = os.path.join(path, 'mt-metrics-eval')
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
    tar.extractall(path)


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
