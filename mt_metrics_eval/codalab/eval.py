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
#!/usr/bin/env python3

"""Script to score codalab submissions to PHASE 1 (open) at WMT Metrics 2023.

Usage: eval.py INPUT OUTPUT

INPUT and OUTPUT are directories assumed to be structured as follows:
  INPUT/ref - subdirectory containing reference files:
    SEG_REF_FILE - segment-level reference scores
    SYS_REF_FILE - system-level reference scores
    REF_FREE_SEG_REF_FILE - segment-level reference scores for QE metrics
    REF_FREE_SYS_REF_FILE - system-level reference scores for QE metrics
  INPUT/res - subdirectory containing submitted files matching the pattern:
    *SEG_RES_SUFFIX - segment-level metric scores
    *SYS_RES_SUFFIX - system-level metric scores
    META_FILE - meta info file
  OUTPUT/OUT_FILE - where results get written

The filenames are defined as global variables below. See comments above the
definitions for details about expected formats.

Assumptions and limitations:

- Accepts one system-level score file, or one segment-level score file, or
  both.
- Score files are identified by the suffixes defined below. The rest of the
  filename is ignored; the metric name is read from the file contents.
- Each submission must contain only one metric name, the same name for system-
  and segment-level scores if both are provided.
- Each submission may be either reference-free or reference-based. Reference-
  free submissions are indicated by 'src' values in the reference field. If a
  submission is reference-free, it must be reference-free for all language pairs
  and granularities.
- Currently, only entries pertaining to the offical language pairs, test-set,
  and reference, are read and checked in detail. Only basic checks are performed
  on other entries for the standard test set. Challenge-set entries are ignored.
- Error checking ensures that the submission is consistent with information
  from the reference score files (system names, number of segments, etc). The
  program throws an exception when the first error is encountered.
- The segment-level reference score files are assumed to contain scores for all
  segments. In general, this will not be the case for actual gold scores; this
  program is currently intended for the initial submission phase only.
- Unlike the submission files, reference score files are allowed to have
  multiple metric names, in order to accommodate different gold score names for
  different language pairs (eg, MQM or DA scores). But there can be at most one
  name per LP.
- System-level Pearson correlations between reference and submission scores are
  written to the output file. If no system-level scores are provided, these are
  derived from averaged segment-level scores. The system-level scores pertain to
  the 'all' (whole-test-set) domain only.
- Segment-level Kendall correlations between reference and submission scores are
  written to the output file. If no segment-level scores are provided, these are
  just 0. The correlations are computed by flattening system x segment scores
  into single vectors.
"""

import collections
import dataclasses
import os
import sys
import numpy as np
import scipy.stats

# Globals

# Files containing pseudo-reference scores, at segment and system granularity,
# in standard metrics submission format. See https://wmt-metrics-task.github.io
SEG_REF_FILE = 'goldlabels.seg.score'
SYS_REF_FILE = 'goldlabels.sys.score'
REF_FREE_SEG_REF_FILE = 'goldlabels.reffree.seg.score'
REF_FREE_SYS_REF_FILE = 'goldlabels.reffree.sys.score'

# Suffixes of the files containing metric scores, in standard metrics
# submission format. The prefix is an arbitrary name for the submitted metric.
SEG_RES_SUFFIX = '.seg.score'
SYS_RES_SUFFIX = '.sys.score'

# Meta-info filename for current submission. If the file does not exist, no
# metadata will be read. Otherwise, it must contain at least the following
# entries (any other etries are ignored):
# team: NAME
# primary: Y[es]|N[o]
META_FILE = 'metadata.txt'

# Place to write results. Fields written (last two are expected by codalab,
# team and primary only written if META_FILE was found):
# - team: NAME
# - metric_name: NAME
# - ref_less: Y|N
# - primary: Y|N
# - LP_pearson: PEARSON
# - LP_kendalltau: KENDALL
# where LP is one of zhen, heen or ende (note, no hyphen).
OUT_FILE = 'scores.txt'

# Map official language pairs to standard references.
LANG_PAIR_TO_REF = {
    'en-de': 'refA',
    'he-en': 'refA',
    'zh-en': 'refA'
}

# Test set we're reading.
TEST_SET = 'generaltest2023'

# Name for the global domain.
GLOBAL_DOMAIN = 'all'


@dataclasses.dataclass
class BasicInfo:
  """Collection of basic test set info for a given language pair."""

  # pylint: disable=g-bare-generic
  domains: set = dataclasses.field(default_factory=set)
  docs: set = dataclasses.field(default_factory=set)
  refs: set = dataclasses.field(default_factory=set)
  systems: set = dataclasses.field(default_factory=set)
  num_segs: int = 0

  def add(self, testset, domain, doc, ref, sysname, segno) -> None:
    if testset != TEST_SET:
      return
    self.domains.add(domain)
    self.docs.add(doc)
    self.refs.add(ref)
    self.systems.add(sysname)
    if segno is not None:
      self.num_segs = max(int(segno), self.num_segs)

  def check(self, ref, lp) -> bool:
    """Check contents against reference info."""
    if self.domains != ref.domains:
      raise ValueError(
          f'{lp} domains don\'t match std: {self.domains} vs {ref.domains}')
    if self.docs != ref.docs:
      raise ValueError(
          f'{lp} documents don\'t match standard: {self.docs} vs {ref.docs}')
    if self.refs != ref.refs:
      raise ValueError(
          f'{lp} references don\'t match standard: {self.refs} vs {ref.refs}')
    if self.num_segs != ref.num_segs:
      raise ValueError(
          f'{lp} segment count doesn\'t match standard: '
          f'{self.num_segs} vs {ref.num_segs}')
    return True


def read_metadata(filename: str, required_keys_only=True):
  """Read and check metadata file."""

  metadata = {}
  if os.path.exists(filename):
    required = {'team', 'primary'}
    with open(filename) as f:
      for line in f:
        line = line.strip()
        if not line: continue
        k, v = line.split(maxsplit=1)
        if k.endswith(':'):
          k = k[:-1]
        k = k.lower()
        if required_keys_only and k not in required:
          continue
        metadata[k] = v

    missing = [k for k in required if k not in metadata]
    if missing:
      missing = ', '.join(f'"{k}"' for k in missing)
      raise ValueError(f'Missing entries in {META_FILE}: {missing}')

    for k in ['primary']:
      if metadata[k].lower() in ['y', 'yes']:
        metadata[k] = 'Y'
      elif metadata[k].lower() in ['n', 'no']:
        metadata[k] = 'N'
      else:
        raise ValueError(f'Value for "{k}" must be Y or N')

    primary = metadata['primary'] == 'Y'
    primary_msg = f'{"" if primary else "non-"}primary submission'
    print(f'Read metadata from {META_FILE} - {primary_msg}')
  else:
    print(f'{META_FILE} not found')

  return metadata


def get_result_filenames(res_dir):
  """Find and check result file names for this submission."""
  submitted_files = os.listdir(res_dir)
  seg_level = [f for f in submitted_files if f.endswith(SEG_RES_SUFFIX)]
  sys_level = [f for f in submitted_files if f.endswith(SYS_RES_SUFFIX)]
  if len(seg_level) > 1 or len(sys_level) > 1:
    raise ValueError(
        'Submission has multiple system- or segment-level score files')
  elif not seg_level and not sys_level:
    raise ValueError(
        f'At least one of METRIC{SEG_RES_SUFFIX} or METRIC{SYS_RES_SUFFIX} '
        'must be supplied.')
  seg_res_file = seg_level[0] if seg_level else None
  sys_res_file = sys_level[0] if sys_level else None
  return seg_res_file, sys_res_file


def in_scope(lp, ref, sysname, testset):
  """Return True if a score-file entry with these attributes is in scope."""
  return (lp in LANG_PAIR_TO_REF and
          ref in {LANG_PAIR_TO_REF[lp], 'src'} and
          ref != sysname and
          testset == TEST_SET)


def read_seg_scores(filename: str):
  """Read and check standard-format segment-level scores."""

  scores = {}  # lp -> sys -> seg -> score
  metrics = {}  # lp -> metric-name, ref
  infos = collections.defaultdict(BasicInfo)  # lp -> BasicInfo
  with open(filename) as f:
    for line in f:
      fields = line.strip().split('\t')
      if len(fields) != 9:
        raise ValueError(f'Expecting 9 tab-separated fields: {line}')
      metric, lp, testset, domain, doc, ref, sysname, segno, score = fields
      infos[lp].add(testset, domain, doc, ref, sysname, segno)
      if not in_scope(lp, ref, sysname, testset): continue
      if lp not in scores:
        scores[lp] = {}
        metrics[lp] = (metric, ref)
      if metric != metrics[lp][0]:
        raise ValueError(f'Multiple metric names provided for {lp}')
      if ref != metrics[lp][1]:
        raise ValueError(
            f'Metric has both source- and reference-baed versions for {lp}')
      if sysname not in scores[lp]:
        scores[lp][sysname] = {}
      segno = int(segno) - 1  # Original numbers are 1-based
      if segno in scores[lp][sysname]:
        raise ValueError(f'Duplicate segment number in {line}: {segno + 1}')
      scores[lp][sysname][segno] = float(score)

  # Convert to np format for convenience
  new_scores = {}
  for lp, syslist in scores.items():
    matrix = []  # system x segment scores
    syslist = sorted(syslist)
    for sysname in syslist:
      num_segs = max(scores[lp][sysname]) + 1
      ordered_scores = [None] * num_segs
      for i, s in scores[lp][sysname].items():
        ordered_scores[i] = s
      if None in ordered_scores:
        m = ordered_scores.count(None)
        raise ValueError(f'Missing {m} segment score(s) for {lp}/{sysname}')
      matrix.append(ordered_scores)
      if len(matrix[-1]) != len(matrix[0]):
        raise ValueError(f'Length mismatch for {lp}/{sysname} segment scores')
    new_scores[lp] = metrics[lp], syslist, np.array(matrix)

  # Return lp -> ((metric, ref), syslist, sys_x_seg score matrix)
  # NB: syslist corresponds to matrix rows, and is sorted so as to be comparable
  # across different score files.
  return new_scores, infos


def read_sys_scores(filename: str):
  """Read and check standard-format system-level scores."""

  scores = {}  # lp -> sys -> domain -> score
  metrics = {}  # lp -> metric name, ref
  infos = collections.defaultdict(BasicInfo)  # lp -> BasicInfo
  with open(filename) as f:
    for line in f:
      fields = line.strip().split('\t')
      if len(fields) != 7:
        raise ValueError(f'Expecting 7 tab-separated fields: {line}')
      metric, lp, testset, domain, ref, sysname, score = fields
      infos[lp].add(testset, domain, None, ref, sysname, None)
      if not in_scope(lp, ref, sysname, testset): continue
      if lp not in scores:
        scores[lp] = {}
        metrics[lp] = (metric, ref)
      if metric != metrics[lp][0]:
        raise ValueError(f'Multiple metric names provided for {lp}')
      if ref != metrics[lp][1]:
        raise ValueError(
            f'Metric has both source- and reference-based versions for {lp}')
      if sysname not in scores[lp]:
        scores[lp][sysname] = {}
      if domain in scores[lp][sysname]:
        raise ValueError(f'Duplicate domain in {line}: {domain}')
      scores[lp][sysname][domain] = float(score)

  # Convert to np format for convenience
  new_scores = {}
  for lp, syslist in scores.items():
    matrix = []  # system x domain scores
    syslist, domainlist = sorted(syslist), None
    for sysname in syslist:
      if domainlist is None:
        domainlist = sorted(scores[lp][sysname])
      elif domainlist != sorted(scores[lp][sysname]):
        raise ValueError(f'Mismatched domains in {lp}/{sysname}: {domainlist}')
      matrix.append([float(scores[lp][sysname][d]) for d in domainlist])
    new_scores[lp] = metrics[lp], syslist, domainlist, np.array(matrix)

  # Return lp -> ((metric, ref), syslist, domainlist, sys_x_domain score matrix)
  # NB: syslist, domainlist are sorted, so comparable across different score
  # files.
  return new_scores, infos


def check_uniqueness(results_scores):
  """Check uniqueness for metric/reference combinations across languages.
  
  We allow different metrics/refs for different LPs in reference scores (eg,
  DA vs MQM), but require only one metric across all LPs in submissions. The
  metric can use different official references for different langauges, but
  if is reference-free (ref = 'src'), it must be reference-free across all LPs.

  Args:
    results_scores: Return from read_*_scores(), maps lp -> (metric, ref), ...

  Returns:
    metric_name, is_ref_free
  """
  metrics = set(x[0][0] for x in results_scores.values())
  refs = set(x[0][1] for x in results_scores.values())
  if (len(metrics) > 1):
    raise ValueError(f'Found multiple metrics: {metrics}')
  metric = list(metrics)[0]
  if 'src' in refs and len(refs) > 1:
    raise ValueError(
        'Metric has both source- and reference-based segment-level versions')
  return metric, 'src' in refs


def check_coverage(results_scores, primary):
  """Check language-pair coverage for results."""
  if primary:
    if not set(LANG_PAIR_TO_REF).issubset(results_scores):
      raise ValueError(
          f'Primary metrics must provide results for {set(LANG_PAIR_TO_REF)}')
  else:
    pass  # Currently anything goes for non-primary submissions.


def main(argv):
  _, input_dir, output_dir = argv
  ref_dir = os.path.join(input_dir, 'ref')
  res_dir = os.path.join(input_dir, 'res')

  def read_ref_scores(level, ref_free):
    if level == 'seg' and not ref_free:
      return read_seg_scores(os.path.join(ref_dir, SEG_REF_FILE))
    elif level == 'seg'and ref_free:
      return read_seg_scores(os.path.join(ref_dir, REF_FREE_SEG_REF_FILE))
    elif level == 'sys' and not ref_free:
      return read_sys_scores(os.path.join(ref_dir, SYS_REF_FILE))
    elif level == 'sys' and ref_free:
      return read_sys_scores(os.path.join(ref_dir, REF_FREE_SYS_REF_FILE))
    else:
      assert False

  def print_summary(metric, ref_free, res_scores, infos):
    print(f'- Metric is {metric}, ref-{"free" if ref_free else "based"}')
    print(f'- Read scores for official languages: {",".join(res_scores)}')
    others = set(infos) - set(res_scores)
    print(f'- Read scores for other languages: {others if others else "None"}',
          flush=True)

  # Read metadata
  metainfo = read_metadata(
      os.path.join(res_dir, META_FILE), required_keys_only=True)
  primary = 'primary' in metainfo and metainfo['primary'] == 'Y'

  seg_ref_scores, sys_ref_scores = None, None

  # Read and check submission files.
  #
  seg_res_file, sys_res_file = get_result_filenames(res_dir)
  #
  seg_metric, seg_res_scores, seg_ref_free = None, None, None
  sys_metric, sys_res_scores, sys_ref_free = None, None, None
  if seg_res_file:
    print(f'Reading and checking {seg_res_file}:', flush=True)
    seg_res_scores, seg_infos = read_seg_scores(
        os.path.join(res_dir, seg_res_file))
    seg_metric, seg_ref_free = check_uniqueness(seg_res_scores)
    check_coverage(seg_res_scores, primary)
    seg_ref_scores, seg_ref_infos = read_ref_scores('seg', seg_ref_free)
    sys_ref_scores, sys_ref_infos = read_ref_scores('sys', seg_ref_free)
    for lp, (_, syslist, matrix) in seg_res_scores.items():
      if syslist != seg_ref_scores[lp][1]:
        raise ValueError(f'System list for {lp} doesn\'t match reference: '
                         f'{syslist} vs {seg_ref_scores[lp][1]}')
      num_segs = matrix.shape[1]
      if num_segs != seg_res_scores[lp][2].shape[1]:
        raise ValueError(f'Num segments for {lp} doesn\'t match reference '
                         f'{num_segs} vs {seg_res_scores[lp][2].shape[1]}')
    for lp, info in seg_infos.items():
      if lp not in seg_ref_infos:
        raise ValueError(f'Unknown segment-level language pair: {lp}')
      info.check(seg_ref_infos[lp], lp)
    print_summary(seg_metric, seg_ref_free, seg_res_scores, seg_infos)
  #
  if sys_res_file:
    print(f'Reading and checking {sys_res_file}:', flush=True)
    sys_res_scores, sys_infos = read_sys_scores(
        os.path.join(res_dir, sys_res_file))
    sys_metric, sys_ref_free = check_uniqueness(sys_res_scores)
    check_coverage(sys_res_scores, primary)
    sys_ref_scores, sys_ref_infos = read_ref_scores('sys', sys_ref_free)
    if seg_ref_scores is None:
      seg_ref_scores, seg_ref_infos = read_ref_scores('seg', sys_ref_free)
    for _, _, domainlist, _ in sys_ref_scores.values():
      assert GLOBAL_DOMAIN in domainlist
    for lp, (_, syslist, domainlist, _) in sys_res_scores.items():
      if syslist != sys_ref_scores[lp][1]:
        raise ValueError(f'System list for {lp} doesn\'t match reference: '
                         f'{syslist} vs {sys_ref_scores[lp][1]}')
      # Currently only using GLOBAL_DOMAIN, but ensure we have scores for all
      # domains in the reference.
      if domainlist != sys_ref_scores[lp][2]:
        raise ValueError(f'Domain list for {lp} doesn\'t match reference: '
                         f'{domainlist} vs {sys_ref_scores[lp][2]}')
    for lp, info in sys_infos.items():
      if lp not in sys_ref_infos:
        raise ValueError(f'Unknown system-level language pair: {lp}')
      info.check(sys_ref_infos[lp], lp)
    print_summary(sys_metric, sys_ref_free, sys_res_scores, sys_infos)
  #
  if seg_res_file and sys_res_file:
    if seg_metric != sys_metric:
      raise ValueError(
          f'System/segment metric name mismatch: {sys_metric} vs {seg_metric}')
    if seg_ref_free != sys_ref_free:
      raise ValueError(
          f'System/segment ref-free mismatch: {sys_ref_free} vs {seg_ref_free}')
  metric_name = seg_metric if seg_res_file else sys_metric
  ref_free = seg_ref_free if seg_res_file else sys_ref_free

  # Create sys-level scores by averaging segment-level scores if no sys-level
  # scores supplied.
  if not sys_res_file:
    print('No system-level scores supplied - averaging segment-level scores')
    sys_res_scores = {}
    for lp, (metric, syslist, matrix) in seg_res_scores.items():
      sys_res_scores[lp] = (
          metric, syslist, [GLOBAL_DOMAIN], matrix.mean(axis=1, keepdims=True))

  # Compute results

  def make_key(lp, corr):
    lp = lp.replace('-', '')
    return f'{lp}_{corr}'

  results = {}
  print('Computing system-level Pearson correlations with pseudo gold scores')
  for lp, (_, _, domainlist, matrix) in sys_res_scores.items():
    _, _, ref_domainlist, ref_matrix = sys_ref_scores[lp]
    scores = matrix[:, domainlist.index(GLOBAL_DOMAIN)]
    ref_scores = ref_matrix[:, ref_domainlist.index(GLOBAL_DOMAIN)]
    results[make_key(lp, 'pearson')] = scipy.stats.pearsonr(
        scores, ref_scores)[0]

  if seg_res_file:
    print('Computing seg-level Kendall correlations with pseudo gold scores')
    for lp, (_, _, matrix) in seg_res_scores.items():
      _, _, ref_matrix = seg_ref_scores[lp]
      scores, ref_scores = matrix.flatten(), ref_matrix.flatten()
      results[make_key(lp, 'kendalltau')] = scipy.stats.kendalltau(
          scores, ref_scores)[0]
  else:
    for lp in seg_ref_scores:
      results[make_key(lp, 'kendalltau')] = 0

  # Write results
  with open(os.path.join(output_dir, OUT_FILE), 'w') as f:
    for k, v in metainfo.items():
      f.write(f'{k}: {v}\n')
    f.write(f'metric_name: {metric_name}\n')
    f.write(f'ref_less: {"Y" if ref_free else "N"}\n')
    for c, s in results.items():
      f.write(f'{c}: {s:f}\n')

# Run
if __name__ == '__main__':
  main(sys.argv)
