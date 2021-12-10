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
r"""Command-line interface for mt-metrics-eval.

List info, print text, score metrics, compare metrics.

Examples (omitting path to binary):

# Info about test sets:
mtme --list  # all available sets
mtme --list -t wmt20  # language pairs for wmt20
mtme --list -t wmt20 -l en-de  # details for wmt20/en-de

# Generate all system outputs, pasted with doc-ids, source, and reference:
mtme -t wmt20 -l en-de --echosys doc,src,ref

# Score a metric in a file containing 'system-name score' entries:
mtme -t wmt20 -l en-de < my-metric.score

# Scoring using alternative gold scores, with outlier systems included:
mtme -t wmt20 -l en-de -g mqm --use_outliers < my-metric.score

# Compare two metrics, testing whether correlations are significantly different:
mtme -t wmt20 -l en-de -g mqm --use_outliers -i metric1.score -c metric2.score
"""

import sys
from absl import app
from absl import flags
from mt_metrics_eval import data
from mt_metrics_eval import stats
import scipy.stats
import glob

flags.DEFINE_bool(
    'download', False, 'Download local copy of the database and quit. '
    'Overwrites any existing copy.')
flags.DEFINE_bool(
    'list', False, 'List available test sets. With -t, list language pairs for '
    'given test set. With -t and -l, list details for given test '
    'set and language pair.')
flags.DEFINE_string(
    'echo', None,
    'A comma-separated list of text names, any of doc, src, ref, or '
    'any alternative reference name (see --list). Pastes the '
    'corresponding texts to STDOUT then quits.')
flags.DEFINE_string(
    'echosys', None,
    'Like --echo, but repeats output once for each system, with "sysname txt " '
    'fields prepended.')
flags.DEFINE_string(
    'test_set', None, 'Test set to use (see --list).', short_name='t')
flags.DEFINE_string(
    'language_pair',
    None,
    'Source-target language pair (2-char ISO639-1 codes).',
    short_name='l')
flags.DEFINE_string(
    'input',
    None, 'Read input from a file instead of STDIN. Each line should '
    'contain a system name and a score, separated by a tab. '
    'The number of entries per system determines granularity: '
    'one per system, document, or segment in the test set. Document and '
    'segment scores must be ordered the same as in the test set.',
    short_name='i')
flags.DEFINE_string(
    'output', None, 'Output file, defaults to STDOUT.', short_name='o')
flags.DEFINE_string(
    'compare',
    None,
    'File containing scores for comparison to --input scores, in the same '
    'format. Comparison can be slow due to resampled significance tests for '
    'document- and segment-level scores. Set --k=1 to disable resampling.',
    short_name='c')
flags.DEFINE_string(
    'gold',
    'std', 'Type of gold scores to compare to, use "std" to designate official '
    'gold scores.',
    short_name='g')
flags.DEFINE_bool(
    'avg', False,
    'Use averaged rather than pooled correlations for doc and seg level '
    'scores.')
flags.DEFINE_integer(
    'k', 1000, 'Number of resampling runs for PERM-BOTH significance test.')
flags.DEFINE_float(
    'thresh', -1, 'Threshold for WMT Kendall-like correlation. Defaults to 25 '
    'if gold scores are WMT raw, otherwise 0.')
flags.DEFINE_bool(
    'use_outliers', False,
    'Include scores for outlier systems in correlation. If these scores are '
    'not available in the set selected with -gold, this option has no effect.')
flags.DEFINE_string(
    'add_systems', '',
    'Comma-separated list of systems to add to the default set for '
    'correlation, for instance outlier or human output. These scores must be '
    'available in the set selected with -gold.')

FLAGS = flags.FLAGS


def PrintCorrelation(evs, scorefile, tag, outfile):
  """Read scores from score file, print correlation stats, return values."""

  scores = data.ReadScoreFile(scorefile)
  if not scores:
    raise ValueError('No systems in input file %s' % scorefile)
  num_scores = len(list(scores.values())[0])
  if num_scores == 1:
    level = 'sys'
  elif num_scores == len(evs.docs):
    level = 'doc'
  elif num_scores == len(evs.src):
    level = 'seg'
  else:
    raise ValueError(
        'Number of scores/system (%d) doesn\'t match any known granularity in '
        '%s/%s' % (num_scores, FLAGS.test_set, FLAGS.language_pair))

  gold_scores = evs.Scores(level, FLAGS.gold)
  std_scorer = data.DATA[FLAGS.test_set]['std_scorers'][level]
  gold_name = std_scorer if FLAGS.gold == 'std' else FLAGS.gold
  if gold_scores is None:
    raise ValueError('No scores for %s at %s level.' % (FLAGS.gold, level))
  sys_names = set(gold_scores) - evs.human_sys_names
  if not FLAGS.use_outliers:
    sys_names -= evs.outlier_sys_names
  for n in [s for s in FLAGS.add_systems.split(',') if s]:
    if n not in gold_scores:
      raise ValueError(f'No {gold_name} scores for system {n}')
    sys_names.add(n)

  corr = evs.Correlation(level, scores, FLAGS.gold, sys_names)
  pearson = corr.Pearson(FLAGS.avg)
  spearman = corr.Spearman(FLAGS.avg)
  kendall = corr.Kendall(FLAGS.avg)
  # Always average KendallLike, otherwise it's very slow.
  if FLAGS.thresh == -1:
    FLAGS.thresh = 25 if gold_name == 'wmt-raw' else 0
  kendall_like = corr.KendallLike(averaged=True, thresh=FLAGS.thresh)

  corr_type = 'averaging' if FLAGS.avg else 'pooling'
  print(
      f'{tag}{FLAGS.test_set} {FLAGS.language_pair} {level}-level: '
      f'scoring {corr.num_sys}/{len(evs.sys_names)} systems, '
      f'gold={gold_name}, '
      f'{corr_type} {corr.num_items}x{corr.num_sys} scores '
      f'({corr.none_count} None): '
      f'Pearson={pearson[0]:0.3f},p{pearson[1]:0.3f} '
      f'Spearman={spearman[0]:0.3f},p{spearman[1]:0.3f} '
      f'Kendall={kendall[0]:0.3f},p{kendall[1]:0.3f} '
      f'Kendall-like@{FLAGS.thresh:g}={kendall_like[0]:0.3f}',
      file=outfile)

  return corr, pearson, spearman, kendall, kendall_like


def PrintComparison(res_base, res_comp, outfile):
  """Test for difference between correlations, and print results."""
  corr1, pears1, spear1, kend1, kendlike1 = res_base
  corr2, pears2, spear2, kend2, kendlike2 = res_comp
  if corr1.num_items != corr2.num_items:
    raise ValueError('Can\'t compare score files at different granularities.')

  pearson = corr1.GenCorrFunction(scipy.stats.pearsonr, FLAGS.avg)
  spearman = corr1.GenCorrFunction(scipy.stats.spearmanr, FLAGS.avg)
  kendall = corr1.GenCorrFunction(scipy.stats.kendalltau, FLAGS.avg)
  # Always average KendallLike, otherwise it's very slow.
  kendlike = stats.KendallLike(corr1.num_sys, FLAGS.thresh)

  def _SigTest(corr1, corr2, v1, v2, corr_fcn):
    better = v2[0] >= v1[0]
    if not better:
      corr2, corr1 = corr1, corr2
    w = stats.WilliamsSigDiff(corr1, corr2, corr_fcn)
    p = stats.PermutationSigDiff(corr1, corr2, corr_fcn, FLAGS.k)
    return better, w, p

  pear_b, pear_w, pear_p = _SigTest(corr1, corr2, pears1, pears2, pearson)
  sper_b, sper_w, sper_p = _SigTest(corr1, corr2, spear1, spear2, spearman)
  kend_b, kend_w, kend_p = _SigTest(corr1, corr2, kend1, kend2, kendall)
  kl_b, kl_w, kl_p = _SigTest(corr1, corr2, kendlike1, kendlike2, kendlike)

  def _Summary(better, sig_williams, sig_perm):
    s = '2>1,' if better else '1>2,'
    s += f'pWilliams={sig_williams[0]:0.3f},pPERM={sig_perm:0.3f}'
    return s

  print(
      'Pearson:%s Spearman:%s Kendall:%s Kendall-like@%g:%s' %
      (_Summary(pear_b, pear_w, pear_p), _Summary(sper_b, sper_w, sper_p),
       _Summary(kend_b, kend_w, kend_p), FLAGS.thresh, _Summary(
           kl_b, kl_w, kl_p)),
      file=outfile)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if FLAGS.download:
    print('Downloading data into %s' % data.LocalDir())
    data.Download()
    return

  if FLAGS.list:
    if FLAGS.test_set is None:
      print('test-sets:', ' '.join(data.DATA))
    elif FLAGS.language_pair is None:
      print(f'language pairs for {FLAGS.test_set}:',
            ' '.join(data.DATA[FLAGS.test_set]['language_pairs']))
    else:
      evs = data.EvalSet(FLAGS.test_set, FLAGS.language_pair)
      print(
          '%s %s:' % (FLAGS.test_set, FLAGS.language_pair),
          '%d segs, %d docs, %d systems (includes %d outliers + %d human), '
          'outliers: {%s}, human: {%s}, refs: {%s}, gold-scores: {%s}' %
          (len(evs.src), len(evs.docs), len(evs.sys_names),
           len(evs.outlier_sys_names), len(evs.human_sys_names), ','.join(
               evs.outlier_sys_names), ','.join(evs.human_sys_names), ','.join(
                   evs.all_refs), ','.join(evs.human_score_names)))
    return

  if FLAGS.test_set is None:
    raise ValueError('No test_set specified.')
  if FLAGS.language_pair is None:
    raise ValueError('No language_pair specified.')

  evs = data.EvalSet(FLAGS.test_set, FLAGS.language_pair)

  if FLAGS.echo is not None or FLAGS.echosys is not None:
    flag_val = FLAGS.echo or FLAGS.echosys
    texts = []
    for col in flag_val.split(','):
      if col == 'src':
        texts.append(evs.src)
      elif col == 'doc':
        docnames = []
        for d, (b, e) in evs.docs.items():
          docnames += [d] * (e - b)
        texts.append(docnames)
      elif col in evs.all_refs:
        texts.append(evs.all_refs[col])
      else:
        raise ValueError('Unknown text type for --echo: %s' % col)
    if FLAGS.echo is not None:
      for lines in zip(*texts):
        print('\t'.join(lines))
    else:
      for sysname, sysout in evs.sys_outputs.items():
        for lines in zip(sysout, *texts):
          print('%s\t%s' % (sysname, '\t'.join(lines)))
    return

  fh = open(FLAGS.output, 'w') if FLAGS.output else sys.stdout
  with fh:
    tag = '1: ' if FLAGS.compare else ''
    res_base = PrintCorrelation(evs, FLAGS.input or '/dev/stdin', tag, fh)
    if FLAGS.compare:
      res_comp = PrintCorrelation(evs, FLAGS.compare, '2: ', fh)
      PrintComparison(res_base, res_comp, fh)


if __name__ == '__main__':
  app.run(main)
