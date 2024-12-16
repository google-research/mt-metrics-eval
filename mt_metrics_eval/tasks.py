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
"""Define and operate on tasks.

A 'Task' is a collection of attributes that defines a set of metrics that have
been used to score a test set, along with a method for comparing their
performance. Tasks can be run to produce correlation values and clustered rank
assignments resulting from pairwise significance tests over correlations.

A 'TaskSet' is a set of tasks that can be constructed using a compact syntax to
specify different combinations of attributes. TaskSets can be combined to
produce larger sets.

The 'TaskResults' and 'TaskSetResults' classes contain the results of running
Tasks and TaskSets, and provide operations like weighted rank or score averaging
to summarize the performance of different metrics across tasks.
"""

from __future__ import annotations

import dataclasses
import io
import itertools
import json
from typing import Any
from mt_metrics_eval import data
from mt_metrics_eval import meta_info
from mt_metrics_eval import stats
import numpy as np
import scipy.stats
import glob

CORRELATION_FUNCTIONS = {
    'pearson': scipy.stats.pearsonr,
    'kendall': scipy.stats.kendalltau,
    'spearman': scipy.stats.spearmanr,
    'accuracy': None,  # Implicit in CompareMetricsWithGlobalAccuracy
    'KendallLike': stats.KendallLike,
    'KendallVariants': stats.KendallVariants,
    'KendallWithTiesOpt': stats.KendallWithTiesOpt,
    'pce': None,  # Implicit in CompareMetricsWithPairwiseConfidenceError
}


def Attributes():
  """List task attributes in canonical order."""
  return list(Task.__annotations__.keys())


def MatrixString(corr_ranks, matrix, pval=0.05, probs=False):
  """Return a string representation of metric ranking and significance."""
  fh = io.StringIO()
  data.PrintMetricComparison(corr_ranks, matrix, pval, file=fh, probs=probs)
  return fh.getvalue()


def _FormatMetric(basename, status, noref):
  status_str = {'primary': '', 'contrastive': '*', 'baseline': '_'}
  noref_str = {True: '[noref]', False: ''}
  return f'{status_str[status]}{basename}{noref_str[noref]}'


def _TsvTable(metric_infos, rows, headers) -> str:
  """TSV-formatted table, helper for MetricsTable."""
  metrics = [_FormatMetric(m, s, r) for m, s, r in metric_infos]
  ret = ''
  for row in headers:
    ret += '\t'.join(row[:1] + [x + '\t' for x in row[1:-1]] + row[-1:]) + '\n'
  for m, row in zip(metrics, rows):
    row = [m] + ['--\t--' if r is None else f'{r:d}\t{c:f}' for c, r in row]
    ret += '\t'.join(row) + '\n'
  return ret


def _TextTable(metric_infos, rows, headers, rerank) -> str:
  """Text-formatted table, helper for MetricsTable."""

  def FormatCorrRank(corr, rank, rank_last):
    if corr is None: return '--'
    return f'{corr:6.3f} ({rank})' if rank_last else f'{rank}{corr:6.3f}'

  metrics = [_FormatMetric(m, s, r) for m, s, r in metric_infos]
  row_strs = []
  for m, row in zip(metrics, rows):
    row_str = [FormatCorrRank(c, r, rr) for (c, r), rr in zip(row, rerank)]
    row_strs.append([m] + row_str)
  col_widths = []
  for i in range(len(row_strs[0])):
    rjust = 1 if i > 0 and not rerank[i - 1] else -1
    col_widths.append(rjust * max(len(s[i]) for s in headers + row_strs))
  ret = ''
  def _Just(s, w):
    return f'{s:>{w}}' if w >= 0 else f'{s:<{-w}}'
  for header in headers:
    ret += '  '.join(_Just(h, w) for h, w in zip(header, col_widths)) + '\n'
  if headers:
    ret += '  '.join('-' * abs(w) for w in col_widths) + '\n'
  for row in row_strs:
    ret += '  '.join(_Just(r, w) for r, w in zip(row, col_widths)) + '\n'
  return ret


def _LatexTable(metric_infos, rows, headers, rerank) -> str:
  """Latex-formatted table, helper for MetricsTable."""

  def FormatCorrRank(corr, rank, rank_last):
    if corr is None: return '-- & --'
    bold = rank == 1
    corr = f'{corr:0.3f}'
    rank = f'({rank:d})' if rank_last else f'{rank:d}'
    if bold:
      corr, rank = f'\\textbf{{{corr}}}', f'\\textbf{{{rank}}}'
    return f'{corr} & {rank}' if rank_last else f'{rank} & {corr}'

  def Esc(s):
    return s.replace('_', '\\_')

  metrics = []
  for metric, status, noref in metric_infos:
    metric = Esc(metric)
    star = '*' if noref else ''
    if status == 'primary':
      metric = f'{metric}{star}'
    elif status == 'baseline':
      metric = f'\\underline{{{metric}}}{star}'
    elif status == 'contrastive':
      metric = f'\\textit{{{metric}}}{star}'
    else:
      metric = f'{metric}{star}'
    metrics.append(metric)

  sep, eol = ' & ', ' \\\\'
  ret = ''
  ret += '\\begin{tabular}{l' + '|rr' * len(rows[0]) + '}\n\\toprule\n'
  for h in headers:
    h = [Esc(x) for x in h]
    ret += h[0] + sep + sep.join(
        [f'\\multicolumn{{2}}{{|l}}{{{x}}}' for x in h[1:]])
    ret += eol + '\n'
  if headers: ret += '\\midrule\n'
  for metric, row in zip(metrics, rows):
    row_str = [FormatCorrRank(c, r, rr) for (c, r), rr in zip(row, rerank)]
    ret += f'{metric} & ' + ' & '.join(row_str) +  eol + '\n'
  ret += '\\bottomrule\n\\end{tabular}\n'
  return ret


def MetricsTable(
    metrics: list[str],
    columns: list[dict[str, tuple[float, int]]],
    column_headers: list[list[str]],
    fmt: str = 'text',
    which_metrics: str = 'listed',
    rerank: list[bool] | None = None,
    baselines_metainfo: meta_info.MetaInfo | None = None,
) -> str:
  """Generate a tabular string representation for a set of metric stats.

  Args:
    metrics: A list of metric names that determines the order in which metrics
      are displayed in the leftmost column of the table.
    columns: Mappings from metric name to (correlation, rank) values, the
      content to display in the second and subsequent columns in the table.
    column_headers: A list of 0 or more rows to use as column headers. Each row
      must contain headers for the first metric-name column, followed by each
      column in the columns arg. Header rows are stacked vertically.
    fmt: Format to use for the table: one of 'tsv', 'text', or 'latex'. The
      latex format produces a tabular entity that uses top/mid/bottomrule
      commands from the booktabs package.
    which_metrics: Determines the set of metrics that appear in the leftmost
      column of the table. If 'listed', exactly those in the metrics arg; if
      'intersection', the intersection of the metrics arg with the metrics in
      all columns; if 'union', the union of the metrics arg with the metrics in
      all columns. Metrics not in the metrics arg are placed in alphabetical
      order at the bottom of the table.
    rerank: For each column, display the rank order of each metric relative to
      the other metrics in the column instead of the metrics' original ranks
      from the columns arg. These can differ if the original ranks indicate
      significance clusters or if they pertain to a larger set of metrics. This
      option also triggers a different display if fmt is 'text' or 'latex':
      "corr (rank)" versus the default "rank corr". A None value sets rerank to
      False for all columns.
    baselines_metainfo: MetaInfo object used to determine whether a metric is a
      baseline for display purposes. If None, baseline status of metrics isn't
      indicated.

  Returns:
    A string representation of the data in the desired format.
  """
  # Checks
  assert metrics
  assert columns
  for h in column_headers:
    assert len(h) == len(columns) + 1
  assert fmt in {'tsv', 'text', 'latex'}, fmt
  assert which_metrics in {'listed', 'intersection', 'union'}, which_metrics
  if rerank:
    assert len(rerank) == len(columns)

  if not rerank:
    rerank = [False] * len(columns)

  # Build ordered list of metrics to show results for.
  metrics = metrics.copy()
  if which_metrics == 'intersection':
    metrics_set = set.intersection(*[set(metrics)] + [set(c) for c in columns])
    metrics = [m for m in metrics if m in metrics_set]
  elif which_metrics == 'union':  # Append metrics not in given list
    metrics_set = set.union(*[set(metrics)] + [set(c) for c in columns])
    metrics += sorted([m for m in metrics_set if m not in metrics])

  # Sort ranks in columns if called for.
  new_columns = []
  for rr, col in zip(rerank, columns):
    if rr:
      col = sorted(
          (x for x in col.items() if x[0] in metrics), key=lambda x: -x[1][0])
      col = {m: (c, r + 1) for r, (m, (c, _)) in enumerate(iterable=col)}
    new_columns.append(col)
  columns = new_columns

  # Extract meta-info from metrics list
  metric_infos = []
  for metric in metrics:
    if metric.endswith('[noref]'):
      metric, noref = metric[:-len('[noref]')], True
    else:
      noref = False
    if metric.startswith('*'):
      metric, status = metric[1:], 'contrastive'
    else:
      status = 'primary'
      if baselines_metainfo and metric in baselines_metainfo.baseline_metrics:
        status = 'baseline'
    metric_infos.append((metric, status, noref))

  # Convert columns to rows with None's for missing entries
  rows = []
  for m in metrics:
    rows.append([c[m] if m in c else (None, None) for c in columns])

  if fmt == 'tsv':
    return _TsvTable(metric_infos, rows, column_headers)
  elif fmt == 'text':
    return _TextTable(metric_infos, rows, column_headers, rerank)
  elif fmt == 'latex':
    return _LatexTable(metric_infos, rows, column_headers, rerank)
  else:  # keep lint happy
    return ''


@dataclasses.dataclass()
class Task:
  """Parameters for data.GetCorrelations and data.CompareMetrics*."""
  test_set: str = 'wmt22'
  lang: str = 'en-de'
  domain: str | None = None
  level: str = 'sys'
  human: bool = True
  avg_by: str = 'none'
  corr_fcn: str = 'pearson'
  k: int = 1000
  # None selects standard values for the following three parameters.
  gold: list[str] | str | None = None
  refs: list[set[str]] | set[str] | None = None
  close_refs: list[set[str]] | set[str] | None = None
  use_outliers: bool = False
  primary: bool = True
  pval: float = 0.05
  block_size: int = 100
  early_min: float = 0.02
  early_max: float = 0.50
  replace_nans_with_zeros: bool = False
  perm_test: str = 'scores'
  corr_fcn_args: dict[str, Any] | None = None

  def _StdGold(self, lang, level):
    return meta_info.DATA[self.test_set][lang].std_gold[level]

  def _StdRefs(self, lang):
    return {meta_info.DATA[self.test_set][lang].std_ref}

  def __post_init__(self):
    """Check and fill in some default values."""
    test_set, lang, level = self.test_set, self.lang, self.level
    assert test_set in meta_info.DATA
    assert self.corr_fcn in CORRELATION_FUNCTIONS

    sub_langs = lang.split(',')
    if self.corr_fcn == 'accuracy':
      # Special case: system-level accuracy over multiple language pairs.
      assert self.level == 'sys'
      for sl in sub_langs:
        assert sl in meta_info.DATA[test_set], sl
      if self.gold is None:
        self.gold = [self._StdGold(sl, level) for sl in sub_langs]
      elif not isinstance(self.gold, list):
        self.gold = [self.gold]
      if self.refs is None:
        self.refs = [self._StdRefs(sl) for sl in sub_langs]
      elif not isinstance(self.refs, list):
        self.refs = [self.refs]
      if self.close_refs is None:
        self.close_refs = [set() for _ in sub_langs]
      elif not isinstance(self.close_refs, list):
        self.close_refs = [self.close_refs]
      assert len(self.gold) == len(sub_langs)
      assert len(self.refs) == len(sub_langs)
      assert len(self.close_refs) == len(sub_langs)
    else:
      # Standard correlation over single language pair.
      assert len(sub_langs) == 1
      assert lang in meta_info.DATA[test_set], lang
      assert level in meta_info.DATA[test_set][lang].std_gold, level
      if self.gold is None:
        self.gold = self._StdGold(lang, level)
      if self.refs is None:
        self.refs = self._StdRefs(lang)
      if self.close_refs is None:
        self.close_refs = set()
      assert isinstance(self.gold, str)
      assert isinstance(self.refs, set)
      assert isinstance(self.close_refs, set)
    if self.corr_fcn_args is None:
      self.corr_fcn_args = {}
    # Canonical order for comparisons.
    self.corr_fcn_args = dict(sorted(self.corr_fcn_args.items()))

  @property
  def name(self):
    """Single string attr=value representation."""
    return ' '.join(f'{a}={self.StrVal(a)}' for a in Attributes())

  def StrVal(self, attr):
    return f'{getattr(self, attr)}'.replace(' ', '')

  def Run(self, eval_set_dict=None, parallel_file=None) -> TaskResults:
    """Generate metric correlations and pairwise significance results."""

    def _Evs(lp):
      if eval_set_dict is None:
        return data.EvalSet(self.test_set, lp, read_stored_metric_scores=True)
      else:
        return eval_set_dict[(self.test_set, lp)]

    psd = stats.PermutationSigDiffParams(
        self.block_size, self.early_min, self.early_max)

    if self.corr_fcn == 'accuracy':
      evs_list = [_Evs(lp) for lp in self.lang.split(',')]
      res = data.CompareMetricsWithGlobalAccuracy(
          evs_list, self.refs, self.close_refs, self.human,
          self.use_outliers, self.gold, self.primary,
          self.domain, self.k, psd, self.pval,
          parallel_file=parallel_file)
    elif self.corr_fcn == 'pce':
      # PCE is system-level, but it requires segment-level scores, so the
      # Correlation objects actually require passing "seg" instead of
      # `self.level`.
      corrs = data.GetCorrelations(
          _Evs(self.lang), 'seg', self.refs, self.close_refs, self.human,
          self.use_outliers, self.gold, self.primary, self.domain,
          metric_format='spreadsheet')
      res = data.CompareMetricsWithPairwiseConfidenceError(
          corrs, self.k, psd, self.pval,
          self.replace_nans_with_zeros, self.perm_test,
          parallel_file=parallel_file, **self.corr_fcn_args)
    else:
      corr_fcn = CORRELATION_FUNCTIONS[self.corr_fcn]
      corrs = data.GetCorrelations(
          _Evs(self.lang), self.level, self.refs, self.close_refs, self.human,
          self.use_outliers, self.gold, self.primary, self.domain,
          metric_format='spreadsheet')
      res = data.CompareMetrics(
          corrs, corr_fcn, self.avg_by, self.k, psd, self.pval,
          self.replace_nans_with_zeros, self.perm_test,
          parallel_file=parallel_file, **self.corr_fcn_args)
    return TaskResults(self, res)


class TaskResults:
  """Results from running a Task."""

  def __init__(self, task=None, compare_metrics_results=None):
    """Construct from task and results from CompareMetrics*()."""
    if task:
      self.name, self.pval = task.name, task.pval
    else:
      self.name, self.pval = '', 0

    if compare_metrics_results:
      (
          self.corr_ranks, self.matrix, self.draws_index, self.draws_list
      ) = compare_metrics_results  # type: ignore
    else:
      (
          self.corr_ranks, self.matrix, self.draws_index, self.draws_list
      ) = {}, np.array([]), np.array([]), np.array([])

  def __eq__(self, other):
    return (self.name == other.name and
            self.pval == other.pval and
            self.corr_ranks == other.corr_ranks and
            np.array_equal(self.matrix, other.matrix) and
            np.array_equal(self.draws_index, other.draws_index) and
            np.array_equal(self.draws_list, other.draws_list))

  @property
  def attr_vals(self) -> dict[str, str]:
    """Return attr:val representation of task."""
    return dict(av.split('=') for av in self.name.split())

  @property
  def metrics(self) -> list[str]:
    """Metrics in descending order by correlation."""
    return list(self.corr_ranks.keys())

  @property
  def range(self) -> tuple[float, float]:
    """Return the range of possible scores for this task."""
    corr_fcn = self.attr_vals['corr_fcn']
    if corr_fcn == 'accuracy' or corr_fcn == 'pce' or (
        corr_fcn == 'KendallWithTiesOpt' and
        "'variant':'23'" not in self.attr_vals['corr_fcn_args']):
      return (0, 1)  # accuracy
    else:
      return (-1, 1)  # correlation

  def Corr(self, metric) -> float:
    """Correlation for metric (by name or index)."""
    if isinstance(metric, int): metric = self.metrics[metric]
    return self.corr_ranks[metric][0]

  def Rank(self, metric: str) -> int:
    """Cluster rank for index (by name or index)."""
    if isinstance(metric, int): metric = self.metrics[metric]
    return self.corr_ranks[metric][1]

  def Sig(self, m1, m2) -> bool:
    """Corr(m1) - Corr(m2) is significant. Difference assumed to be >= 0."""
    if isinstance(m1, str): m1 = self.metrics.index(m1)
    if isinstance(m2, str): m2 = self.metrics.index(m2)
    return self.matrix[m1, m2] < self.pval

  def Draws(self, m1, m2) -> np.ndarray:
    """List of resampling draws for m1 versus m2."""
    if self.draws_index is None or len(self.draws_index) == 0:
      return np.array([])
    if isinstance(m1, str): m1 = self.metrics.index(m1)
    if isinstance(m2, str): m2 = self.metrics.index(m2)
    if m1 < m2:
      beg, end = self.draws_index[m1, m2], self.draws_index[m2, m1]
      return self.draws_list[beg:end]
    else:
      # Preserve convention that each pair of draws pertains to (m1, m2)
      beg, end = self.draws_index[m2, m1], self.draws_index[m1, m2]
      return self.draws_list[beg:end][:, [1, 0]]

  def Str(self, probs=False):
    """Return a string representation of metric ranking and significance."""
    return MatrixString(
        self.corr_ranks, self.matrix, self.pval, probs=probs)

  def Save(self, filename):
    """Save results to filename.json and filename.npz."""

    with open(f'{filename}.json', 'w') as f:
      elems = (self.name, self.pval, self.corr_ranks, self.matrix.tolist())
      json.dump(elems, f)
    with open(f'{filename}.npz', 'wb') as f:
      np.savez_compressed(
          f, draws_index=self.draws_index, draws_list=self.draws_list)

  def Load(self, filename):
    """Load results previously saved to filename."""
    with open(f'{filename}.json') as f:
      name, pval, corr_ranks, matrix = json.load(f)
    self.name = name
    self.pval = pval
    self.corr_ranks = corr_ranks
    self.matrix = np.asarray(matrix)
    with open(f'{filename}.npz', 'rb') as f:
      a = np.load(f)
      self.draws_index = a['draws_index']
      self.draws_list = a['draws_list']
    return self


class TaskSet:
  """Convenience class to create and operate on sets of tasks."""

  def __init__(
      self, attr_combs: dict[str, list[Any]] | None = None, **attrs):
    """Construct with given attribute/value combinations.

    Args:
      attr_combs: Dictionary mapping attributes to lists of values. One Task
        will be created for each complete attribute/value combination (zero
        tasks if attr_combs is None).
      **attrs: Remaining attribute/value pairs to pass to the Task constructor,
        for each task specified by attr_combs.
    """
    self.tasks = []
    self.eval_set_dict = {}  # Lazily set by Run.
    if not attr_combs: return
    for vals in itertools.product(*attr_combs.values()):
      comb = dict(zip(attr_combs.keys(), vals))
      self.tasks.append(Task(**comb, **attrs))

  def _BuildEvalSetDict(self):
    for task in self.tasks:
      for lang in task.lang.split(','):
        if (task.test_set, lang) not in self.eval_set_dict:
          self.eval_set_dict[(task.test_set, lang)] = data.EvalSet(
              task.test_set, lang, True)

  def __len__(self):
    return len(self.tasks)

  def __add__(self, other):
    """Combine tasks sets. Any duplicate tasks will get repeated."""
    res = TaskSet()
    res.tasks = self.tasks + other.tasks
    res.eval_set_dict = {**self.eval_set_dict, **other.eval_set_dict}
    return res

  def __iter__(self):
    return iter(self.tasks)

  def Append(self, task: Task):
    self.tasks.append(task)

  def Run(self,
          eval_set_dict: dict[tuple[str, str], data.EvalSet] | None = None
          ) -> TaskSetResults:
    """Run all tasks.

    Args:
      eval_set_dict: Maps (test-set, lp) pairs to EvalSets. This can be used to
        modify EvalSets, for instance by controlling the metrics that will be
        evaluated using AddMetricsFromDir() or AddMetric(). Any (test-set, lp)
        combinations missing from eval_set_dict will be added automatically.

    Returns:
      TaskSetResults object containing results of this run.    
    """
    if eval_set_dict:
      self.eval_set_dict = eval_set_dict.copy()
    self._BuildEvalSetDict()
    return TaskSetResults([task.Run(self.eval_set_dict) for task in self.tasks])


class TaskSetResults:
  """Operations on results from running a TaskSet."""

  def __init__(self, results: list[TaskResults]):
    self.results = results

  def __len__(self):
    return len(self.results)

  def __add__(self, other):
    return TaskSetResults(self.results + other.results)

  def __iter__(self):
    return iter(self.results)

  def Append(self, result: TaskResults):
    self.results.append(result)

  def SplitByAttr(self, attr: str) -> dict[str, TaskSetResults]:
    """Partition into subsets by values of an attribute."""
    subsets = {}
    for result in self.results:
      val = result.attr_vals[attr]
      if val not in subsets:
        subsets[val] = TaskSetResults([])
      subsets[val].Append(result)
    return subsets

  def AssignWeights(
      self, attrs: list[str], total_wt: float = 1.0) -> list[float]:
    """Assign weights to tasks.

    This evenly distributes total_wt across values for the first attribute in
    the attrs list, then recurses. If the attribute list is empty, all remaining
    tasks are equally weighted.

    Args:
      attrs: List of attributes in Attributes().
      total_wt: Total weight to be assigned to tasks.

    Returns:
      List of weights for tasks in results, in order. Weights sum to total_wt.
    """
    if not attrs:
      return [total_wt / len(self.results)] * len(self.results)
    weights = {r.name: 0 for r in self.results}
    subsets = self.SplitByAttr(attrs[0]).values()
    for subset in subsets:
      subweights = subset.AssignWeights(attrs[1:], total_wt / len(subsets))
      # Subset isn't necessarily contiguous within self.results.
      for r, w in zip(subset.results, subweights):
        weights[r.name] = w
    return list(weights.values())

  def AverageRanks(self, weights=None) -> dict[str, float]:
    """Return sorted average weighted rank of metrics available in all tasks.

    Args:
      weights: List of weights, as returned by AssignWeights(). If None, use
        uniform weights.

    Returns:
      Map from metric names to average ranks, ordered by increasing rank. Only
      metrics that appear in all tasks are included.
    """
    if weights is None:
      weights = [1 / len(self)] * len(self)
    ranks = {}
    for res, weight in zip(self.results, weights):
      for metric in res.metrics:
        if metric not in ranks:
          ranks[metric] = []
        ranks[metric].append(res.Rank(metric) * weight)
    ranks = {m: sum(ranks[m]) for m in ranks if len(ranks[m]) == len(self)}
    return dict(sorted(ranks.items(), key=lambda x: x[1]))

  def AverageCorrs(self, weights=None) -> dict[str, float]:
    """Return sorted average weighted correlation of metrics over all tasks.

    For consistency between correlations, which return values in [-1, 1] and
    accuracies, which return values in [0, 1], we transform all scores to the
    [0, 1] range before averaging.

    Args:
      weights: List of weights, as returned by AssignWeights(). If None, use
        uniform weights.

    Returns:
      Map from metric names to average correlations, ordered by decreasing
      correlation.
      Only metrics that appear in all tasks are included.
    """
    if weights is None:
      weights = [1 / len(self)] * len(self)
    corrs = {}
    for res, weight in zip(self.results, weights):
      b, e = res.range
      for metric in res.metrics:
        if metric not in corrs:
          corrs[metric] = []
        score = (res.Corr(metric) - b) / (e - b)
        corrs[metric].append(score * weight)
    corrs = {m: sum(corrs[m]) for m in corrs if len(corrs[m]) == len(self)}
    return dict(sorted(corrs.items(), key=lambda x: -x[1]))

  def AverageCorrMatrix(
      self, weights=None, pval=0.05
  ) -> tuple[dict[str, tuple[float, int]], np.ndarray]:
    """Compute a significance matrix over average correlations.

    This first computes the average correlations for metrics that exist in
    all tasks, then performs a pairwise comparison to determine which
    differences are significant. It re-uses the resampling draws for per-task
    pairwise significance tests stored in the draws_index and draws_list members
    of each component TaskResults object.

    Args:
      weights: List of weights, as returned by AssignWeights(). Assumed to be
      normalized. If None, use uniform weights.
      pval: p-value for determining significant differences in scores.

    Returns:
    - Mapping from metric name to (correlation, rank) pairs, where rank is the
      rank of the metric's significance cluster, ordered by decreasing
      average correlation.
    - Significance matrix: a square numpy array whose rows and columns represent
      metrics, sorted to match the keys in the returned correlation map.
      sig_matrix[i, j] contains the p-value for the null hypothesis that the
      correlation for metric j is >= the correlation for metric i. Only
      entries for which j > i are valid.
    """
    if weights is None:
      weights = [1 / len(self)] * len(self)
    avg_corrs = self.AverageCorrs(weights)

    n, metrics = len(avg_corrs), list(avg_corrs)
    sig_matrix = np.zeros((n, n))
    for i in range(n):
      for j in range(i + 1, n):
        m1, m2 = metrics[i], metrics[j]
        corr_diff = avg_corrs[m1] - avg_corrs[m2]

        max_draws = max(len(r.Draws(m1, m2)) for r in self.results)
        assert(max_draws > 0)
        per_task_draws = np.zeros((len(self), max_draws))
        for k, r in enumerate(self.results):
          b, e = r.range
          draws = weights[k] * (r.Draws(m1, m2) - b) / (e - b)
          draws = draws[:, 0] - draws[:, 1]  # r.Draws() handles order changes.
          # This task and metric pair might have fewer draws due to early
          # stopping, so repeat the actual draws to fill up the space.
          num_copies = int(np.ceil(max_draws / len(draws)))
          per_task_draws[k] = np.tile(draws, num_copies)[:max_draws]
        # Summing along the task dimension gives us an overall average diff
        # under the null hyp, which we compare to the observed diff.
        null_diffs = np.sum(per_task_draws, axis=0)
        sig_matrix[i, j] = np.mean(null_diffs >= corr_diff)

    ranks = data.AssignRanks(sig_matrix, pval)
    corrs_and_ranks = {m: (c, r) for (m, c), r in zip(avg_corrs.items(), ranks)}
    return corrs_and_ranks, sig_matrix

  def Table(
      self,
      metrics: list[str] | None = None,
      initial_column: dict[str, tuple[float, int | float]] | None = None,
      initial_column_header: str = '',
      task_indices: list[int] | None = None,
      attr_list: list[str] | None = None,
      nicknames: dict[str, str] | None = None,
      fmt: str = 'text',
      which_metrics: str = 'listed',
      rerank: list[bool] | None = None,
      baselines_metainfo: meta_info.MetaInfo | None = None,
  ) -> str:
    """Generate a tabular string representation for current contents.

    This is a convenience wrapper around MetricsTable. By default it generates
    a table whose rows are metrics and whose columns are correlation and rank
    statistics for each task.

    Args:
      metrics: A list of metric names that determines the order in which metrics
        are displayed in the leftmost column of the table. If not supplied, the
        list is taken from the initial_column arg if present, otherwise from the
        first TaskResults. 
      initial_column: Optional initial column, a mapping from metric names to
        correlation stats and optional ranks.
      initial_column_header: Header for initial column.
      task_indices: Indices of tasks to display, integers in 1..len+1. The
        default is to display all tasks.
      attr_list: Optional attributes to display above the tasks, in successive
        rows of headers. These are always followed by a line showing task ids.
      nicknames: Nicknames for attribute values, eg KendallWithTiesOpt -> acc-t.
      fmt: Format to use for the table: one of 'tsv', 'text', or 'latex'.
      which_metrics: One of 'listed', 'intersection', or 'union'. See doc for
        MetricsTable.
      rerank: Re-rank individual columns, as described in MetricsTable. The
        default is to not re-rank.
      baselines_metainfo: MetaInfo object used to determine whether a metric is
        a baseline for display purposes. If None, baseline status of metrics
        isn't indicated.

    Returns:
      A string representation of the results.
    """
    if not self.results:
      return ''

    columns = []
    if initial_column:
      if not isinstance(list(initial_column.values())[0], tuple):
        vals = sorted((-c, m) for m, c in initial_column.items())
        initial_column = {m: (-c, r + 1) for r, (c, m) in enumerate(vals)}
      columns.append(initial_column)
    if not task_indices:
      task_indices = range(1, len(self.results) + 1)
    for i in task_indices:
      columns.append(self.results[i - 1].corr_ranks)

    if not metrics:
      metrics = list(columns[0])

    headers = []
    def Nn(val):
      return nicknames[val] if nicknames and val in nicknames else val
    if attr_list is None: attr_list = []
    for attr in attr_list:
      row = [f'{attr}:', ''] if initial_column else [f'{attr}:']
      row += [Nn(self.results[i - 1].attr_vals[attr]) for i in task_indices]
      headers.append(row)
    row = ['metric', initial_column_header] if initial_column else ['metric']
    row += [f'task{i}' for i in task_indices]
    headers.append(row)

    return MetricsTable(
        metrics, columns, headers, fmt, which_metrics, rerank,
        baselines_metainfo)


def WMT23(lps: list[str] | None = None, primary=True, k=0, gold=None):
  """Generate the WMT23 task set and associated weight vector."""

  # Not strictly necessary to declare this, because setting human=True will
  # only score human outputs if any are  available, but we want to make the
  # human attribute reflect what actually got used, and also want to avoid
  # having to load the EvalSets at this point to get this info automatically.
  lps_with_multiple_refs = {'en-he', 'he-en'}

  def Add(lp, level, corr_fcn, human, gold, **kw_args):
    tasks.Append(Task(
        'wmt23', lp, level=level, corr_fcn=corr_fcn, human=human, gold=gold,
        primary=primary, k=k, **kw_args))

  if lps is None: lps = ['en-de', 'he-en', 'zh-en']
  lps = sorted(lps)

  tasks = TaskSet()

  # 1st task is pairwise accuracy across all lps.
  Add(','.join(lps), 'sys', 'accuracy',
      human=bool(lps_with_multiple_refs & set(lps)),
      gold=[gold] * len(lps) if gold else None)

  # System- and segment-level Pearson, and segment-level accuracy for all lps.
  for lp in lps:
    human = lp in lps_with_multiple_refs
    Add(lp, 'sys', 'pearson', human, gold)
    Add(lp, 'seg', 'pearson', human, gold)
    Add(lp, 'seg', 'KendallWithTiesOpt', human, gold,
        avg_by='item', perm_test='pairs', corr_fcn_args={'sample_rate': 1.0})

  weights = [len(lps)] + [1] * (len(tasks) - 1)
  weights = [w / sum(weights) for w in weights]

  return tasks, weights


def WMT24OnWMT23(lps: list[str] | None = None, primary=True, k=0, gold=None):
  """Generate the WMT24 task set for WMT23 and associated weight vector."""

  # Not strictly necessary to declare this, because setting human=True will
  # only score human outputs if any are available, but we want to make the
  # human attribute reflect what actually got used, and also want to avoid
  # having to load the EvalSets at this point to get this info automatically.
  lps_with_multiple_refs = {'en-he', 'he-en'}

  def Add(lp, level, corr_fcn, human, gold, **kw_args):
    tasks.Append(Task(
        'wmt23', lp, level=level, corr_fcn=corr_fcn, human=human, gold=gold,
        primary=primary, k=k, **kw_args))

  if lps is None: lps = ['en-de', 'he-en', 'zh-en']
  lps = sorted(lps)

  tasks = TaskSet()

  # For each language pair: PCE at the system-level and accuracy at the
  # segment-level.
  for lp in lps:
    human = lp in lps_with_multiple_refs
    Add(
        lp,
        'sys',
        'pce',
        human=human,
        gold=[gold] * len(lps) if gold else None,
    )
    Add(
        lp,
        'seg',
        'KendallWithTiesOpt',
        human,
        gold,
        avg_by='item',
        perm_test='pairs',
        corr_fcn_args={'sample_rate': 1.0},
    )

  weights = [1] * len(tasks)
  weights = [w / sum(weights) for w in weights]

  return tasks, weights


def WMT24(lps: list[str] | None = None, primary=True, k=0, gold=None):
  """Generate the WMT24 task set associated weight vector."""

  # Not strictly necessary to declare this, because setting human=True will
  # only score human outputs if any are available, but we want to make the
  # human attribute reflect what actually got used, and also want to avoid
  # having to load the EvalSets at this point to get this info automatically.
  lps_with_multiple_refs = {'en-de'}

  def Add(lp, level, corr_fcn, human, gold, **kw_args):
    tasks.Append(Task(
        'wmt24', lp, level=level, corr_fcn=corr_fcn, human=human, gold=gold,
        primary=primary, k=k, **kw_args))

  if lps is None: lps = ['en-de', 'en-es', 'ja-zh']
  lps = sorted(lps)

  tasks = TaskSet()

  # For each language pair: PCE at the system-level and accuracy at the
  # segment-level.
  for lp in lps:
    human = lp in lps_with_multiple_refs
    Add(
        lp,
        'sys',
        'pce',
        human=human,
        gold=[gold] * len(lps) if gold else None,
    )
    Add(
        lp,
        'seg',
        'KendallWithTiesOpt',
        human,
        gold,
        avg_by='item',
        perm_test='pairs',
        corr_fcn_args={'sample_rate': 1.0},
    )

  weights = [1] * len(tasks)
  weights = [w / sum(weights) for w in weights]

  return tasks, weights
