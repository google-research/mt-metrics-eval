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

# Get latest version of database (slow).
mtme --download

# Info about test sets:
mtme --list  # all available sets
mtme --list -t wmt20  # language pairs for wmt20
mtme --list -t wmt20 -l en-de  # details for wmt20/en-de

# Generate all system outputs, pasted with doc-ids, source, and reference:
mtme -t wmt20 -l en-de --echosys doc,src,ref

# Correlations for sys- and seg-level scores (using example files from the
# database):
MTME20=$HOME/.mt-metrics-eval/mt-metrics-eval-v2/wmt20/metric-scores
mtme -t wmt20 -l en-de < $MTME20/en-de/COMET-ref.sys.score
mtme -t wmt20 -l en-de < $MTME20/en-de/COMET-ref.seg.score

# Correlations with alternative gold scores, outlier systems included:
mtme -t wmt20 -l en-de -g mqm --use_outliers < $MTME20/en-de/COMET-ref.sys.score

# Compare two metrics, testing whether correlations are significantly different:
METRIC1=$MTME20/en-de/COMET-ref.sys.score
METRIC2=$MTME20/en-de/BLEU-ref.sys.score
mtme -t wmt20 -l en-de -g mqm -i $METRIC1 -c $METRIC2

# Compare all metrics under specified conditions, writing ranks, correlations,
# and matrix of pair-wise significance values (using small k for demo).
mtme --matrix -t wmt20 -l en-de -g mqm --k 100
"""

import ast
import sys
from absl import app
from absl import flags
from mt_metrics_eval import data
from mt_metrics_eval import meta_info
from mt_metrics_eval import stats
from mt_metrics_eval import tasks
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
    'A comma-separated list of text names, any of "domain", "doc", "src", '
    '"ref" for the main reference, or an actual reference name for any other '
    'reference (see --list). Pastes the corresponding tags or text to STDOUT '
    'then quits.')
flags.DEFINE_string(
    'echosys', None,
    'Like --echo, but repeats output once for each system, with "sysname txt " '
    'fields prepended.')
flags.DEFINE_bool(
    'scores', False,
    'Dump all scores to a tsv file. For each system, write the following '
    'fields for each segment: system-name, domain, doc, seg-id, then '
    'segment-level, doc-level, domain-level, and system-level scores '
    '(whichever are available). Gold scores are written first, followed by '
    'metric scores. None values are written whenever scores aren\'t available '
    'for the given level and/or system.')
flags.DEFINE_string(
    'test_set', None, 'Test set to use (see --list).', short_name='t')
flags.DEFINE_string(
    'language_pair', None,
    'Source-target language pair (2-char ISO639-1 codes).', short_name='l')
flags.DEFINE_string(
    'input',
    None, 'Read input from a file instead of STDIN. Each line should '
    'contain a system name and a score, separated by a tab. '
    'The number of entries per system determines granularity: '
    'one per system, document, or segment in the test set.',
    short_name='i')
flags.DEFINE_string(
    'output', None, 'Output file, defaults to STDOUT.', short_name='o')
flags.DEFINE_string(
    'matrix_save', None, 'File for json/npgz output from --matrix option.')
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
flags.DEFINE_string(
    'avg', 'none',
    'Averaging method for segment- or doc-level correlations: "none" to pool '
    'scores into vectors, "item" to average over correlations of item '
    'vectors, or "sys" to average over correlations of system vectors.')
flags.DEFINE_bool(
    'replace_nans_with_zeros', False,
    'Replace NaNs with 0 instead of discarding them. This will penalize '
    'metrics that produce NaN values because they assign all items the same '
    'score.')
flags.DEFINE_integer(
    'k', 1000, 'Number of resampling runs for PERM-BOTH significance test.')
flags.DEFINE_integer(
    'k_block', 1000,
    'Size of blocks for early stopping checks with PERM-BOTH test. Set to >= k '
    'for no early stopping.')
flags.DEFINE_float(
    'early_min', 0.02,
    'Early stop PERM-BOTH if pval < early_min at current block boundary.')
flags.DEFINE_float(
    'early_max', 0.50,
    'Early stop PERM-BOTH if pval > early_max at current block boundary.')
flags.DEFINE_string(
    'matrix_perm_test', 'scores',
    'Type of permutation test to run, one of "scores" or "pairs". The pairs '
    'test only works with KendallTiesWithOpt correlation, with variant set to '
    '"23" or "acc23".')
flags.DEFINE_float(
    'thresh', -1, 'Threshold for WMT Kendall-like correlation. Defaults to 25 '
    'if gold scores are WMT raw, otherwise 0. (If using --matrix, set '
    '--matrix_corr_args to \'{"thresh": 25}\' for the same effect.)')
flags.DEFINE_bool(
    'use_outliers', False,
    'Include scores for outlier systems in correlation. If these scores are '
    'not available in the set selected with -gold, this option has no effect.')
flags.DEFINE_string(
    'add_systems', '',
    'Comma-separated list of systems to add to the default set for '
    'correlation, for instance outlier or human output. These scores must be '
    'available in the set selected with -gold.')
flags.DEFINE_bool(
    'matrix', False, 'Compute correlations for a set of metrics, and perform '
    'significance tests on their differences. Writes metrics in descending '
    'order by correlation, followed by their rank (may include ties), '
    'correlation, then n significance indictors (x for filler, 1 for sig, 0 '
    'for non) for comparisons between current metric and all n metrics, in the '
    'same order as rows. Flags that affect this operation include all '
    '--matrix_* flags, along with --gold, --avg, --k, --k_block, --early_min, '
    '--early_max, --replace_nans_with_zeros, and --use_outliers.')
flags.DEFINE_string(
    'matrix_parallel', None,
    'Parallelize metric comparisions, and use this value as a temp file name.')
flags.DEFINE_string(
    'matrix_level', 'sys', 'Granularity, one of "sys", "doc" or "seg"')
flags.DEFINE_string(
    'matrix_domain', None,
    'Limit matrix correlations to this domain, no limit if None. The string '
    '"None" is also interpreted as None.')
flags.DEFINE_string(
    'matrix_refs', 'std',
    'Reference(s) to use. Metric variants that use references outside this set '
    'are excluded, as are human outputs that match any of these references. '
    'Use "std" to designate the standard reference.')
flags.DEFINE_string(
    'matrix_close_refs', '',
    'Additional reference(s) to always exclude from human outputs when '
    'matrix_human is True.')
flags.DEFINE_bool(
    'matrix_human', False,
    'Include human outputs in matrix calculation, except for references '
    'specified in matrix_refs and matrix_close_refs.')
flags.DEFINE_bool(
    'matrix_primary', True,
    'Use only primary metric submissions in the matrix.')
flags.DEFINE_float(
    'matrix_pval', 0.05,
    'p-value to use for assigning significance to metric comparisons.')
flags.DEFINE_string(
    'matrix_corr', 'pearson',
    'Correlation to use for --matrix, one of pearson, spearman, kendall, '
    'accuracy, or any of the vector-based correlation functions defined in '
    'the stats module, eg KendallVariants. '
    'Accuracy is valid only for system-level comparisons. It also '
    'triggers special interpretation of the --language_pair, --matrix_refs, '
    'and --matrix_close_refs: language pair can be a comma-separated list, '
    'with corresponding lists of refs or a single ref that gets applied to '
    'all languages (it\'s not possible to specify a set of refs / language '
    'with this option.')
flags.DEFINE_string(
    'matrix_corr_args', '{}',
    'Extra arguments to the matrix_corr function, a string that can be '
    'converted to a python dict, eg \'{"variant": "acc23", "epsilon": 10}\'.')


FLAGS = flags.FLAGS


def PrintScores(evs):
  """Print all scores in tsv format. See doc for --scores option."""

  sys_names = sorted(evs.sys_names)
  gold_names = sorted(evs.human_score_names)
  metric_names = sorted(evs.metric_names)

  header = ''
  fields = ['system-name', 'domain', 'doc', 'seg-id']
  for level in 'seg', 'doc', 'domain', 'sys':
    if level in evs.levels:
      fields += [f'{g}:{level}' for g in gold_names]
      fields += [f'{m}:{level}' for m in metric_names]
    header = '\t'.join(fields) + '\n'
  docs = evs.DocsPerSeg()
  domains = evs.DomainsPerSeg()
  domain_ids = {d: i for i, d in enumerate(evs.domain_names)}
  doc_ids = {d: i for i, d in enumerate(evs.doc_names)}

  def _Score(level, scorer, sysname, ind):
    scores = evs.Scores(level, scorer)
    if scores is None or sysname not in scores:
      return 'None'
    else:
      return f'{scores[sysname][ind]}'

  fh = open(FLAGS.output, 'w') if FLAGS.output else sys.stdout
  with fh:
    fh.write(header)
    for n in sys_names:
      for i in range(len(evs.src)):
        doc, domain = docs[i], domains[i]
        doc_id, domain_id = doc_ids[doc], domain_ids[domain]
        fields = [n, domain, doc, f'{i + 1}']
        for level in 'seg', 'doc', 'domain', 'sys':
          if level not in evs.levels:
            continue
          ind = {'seg': i, 'doc': doc_id, 'domain': domain_id, 'sys': 0}[level]
          fields += [_Score(level, g, n, ind) for g in gold_names]
          fields += [_Score(level, m, n, ind) for m in metric_names]
        fh.write('\t'.join(fields) + '\n')


def Flag2TaskArg(flag_val, sets=False):
  """Convert gold and ref flag values to task arguments."""
  if flag_val == 'std' or flag_val == 'None' or not flag_val:
    return None
  vals = flag_val.split(',')
  if sets:
    # Limited to singleton sets.
    vals = [{v} for v in vals]
  return vals[0] if len(vals) == 1 else vals


def PrintMatrix():
  """Print ranks, correlations, and comparison matrix for a set of metrics."""

  task = tasks.Task(
      test_set=FLAGS.test_set,
      lang=FLAGS.language_pair,
      domain=None if FLAGS.matrix_domain == 'None' else FLAGS.matrix_domain,
      level=FLAGS.matrix_level,
      human=FLAGS.matrix_human,
      avg_by=FLAGS.avg,
      corr_fcn=FLAGS.matrix_corr,
      k=FLAGS.k,
      gold=Flag2TaskArg(FLAGS.gold),
      refs=Flag2TaskArg(FLAGS.matrix_refs, sets=True),
      close_refs=Flag2TaskArg(FLAGS.matrix_close_refs, sets=True),
      use_outliers=FLAGS.use_outliers,
      primary=FLAGS.matrix_primary,
      pval=FLAGS.matrix_pval,
      block_size=FLAGS.k_block,
      early_min=FLAGS.early_min,
      early_max=FLAGS.early_max,
      replace_nans_with_zeros=FLAGS.replace_nans_with_zeros,
      perm_test=FLAGS.matrix_perm_test,
      corr_fcn_args=ast.literal_eval(FLAGS.matrix_corr_args)
  )
  task_results = task.Run(parallel_file=FLAGS.matrix_parallel)
  fh = open(FLAGS.output, 'w') if FLAGS.output else sys.stdout
  with fh:
    fh.write(task_results.name + '\n')
    fh.write(task_results.Str())
  if FLAGS.matrix_save:
    task_results.Save(FLAGS.matrix_save)


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

  std_scorer = evs.StdHumanScoreName(level)
  gold_name = std_scorer if FLAGS.gold == 'std' else FLAGS.gold
  gold_scores = evs.Scores(level, gold_name)
  if gold_scores is None:
    raise ValueError('No scores for %s at %s level.' % (FLAGS.gold, level))
  sys_names = set(gold_scores) - evs.human_sys_names
  if not FLAGS.use_outliers:
    sys_names -= evs.outlier_sys_names
  for n in [s for s in FLAGS.add_systems.split(',') if s]:
    if n not in gold_scores:
      raise ValueError(f'No {gold_name} scores for system {n}')
    sys_names.add(n)

  avg = 'none' if level == 'sys' else FLAGS.avg
  corr = evs.Correlation(gold_scores, scores, sys_names)
  pearson = corr.Pearson(FLAGS.avg)
  spearman = corr.Spearman(FLAGS.avg)
  kendall = corr.Kendall(FLAGS.avg)
  # Always use item-wise averaging with KendallLike, otherwise it's very slow.
  if FLAGS.thresh == -1:
    FLAGS.thresh = 25 if gold_name == 'wmt-raw' else 0
  kendall_like = corr.KendallLike(thresh=FLAGS.thresh)

  if avg == 'none':
    cmp = 'flattened'
  elif avg == 'sys':
    cmp = 'rows in'
  else:
    cmp = 'columns in'

  print(
      f'{tag}{FLAGS.test_set} {FLAGS.language_pair} {level}-level: '
      f'scoring {corr.num_sys}/{len(evs.sys_names)} systems, '
      f'gold={gold_name}, '
      f'comparing {cmp} {corr.num_sys}x{corr.num_items} matrices '
      f'({corr.none_count} None vals): '
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

  pearson = corr1.AverageCorrelation(
      scipy.stats.pearsonr, FLAGS.avg, FLAGS.replace_nans_with_zeros)
  spearman = corr1.AverageCorrelation(
      scipy.stats.spearmanr, FLAGS.avg, FLAGS.replace_nans_with_zeros)
  kendall = corr1.AverageCorrelation(
      scipy.stats.kendalltau, FLAGS.avg, FLAGS.replace_nans_with_zeros)
  # Always average KendallLike, otherwise it's very slow.
  kendlike = corr1.AverageCorrelation(
      stats.KendallLike, 'item', FLAGS.replace_nans_with_zeros,
      thresh=FLAGS.thresh)

  def _SigTest(corr1, corr2, v1, v2, corr_wrapper, corr_fcn):
    better = v2[0] >= v1[0]
    if not better:
      corr2, corr1 = corr1, corr2
    w = stats.WilliamsSigDiff(corr1, corr2, corr_wrapper)
    p, _, _, _ = stats.PermutationSigDiff(
        corr1, corr2, corr_fcn, FLAGS.avg, FLAGS.k,
        stats.PermutationSigDiffParams(
            FLAGS.k_block, FLAGS.early_min, FLAGS.early_max),
        FLAGS.replace_nans_with_zeros)
    return better, w, p

  pear_b, pear_w, pear_p = _SigTest(
      corr1, corr2, pears1, pears2, pearson, scipy.stats.pearsonr)
  sper_b, sper_w, sper_p = _SigTest(
      corr1, corr2, spear1, spear2, spearman, scipy.stats.spearmanr)
  kend_b, kend_w, kend_p = _SigTest(
      corr1, corr2, kend1, kend2, kendall, scipy.stats.kendalltau)
  kl_b, kl_w, kl_p = _SigTest(
      corr1, corr2, kendlike1, kendlike2, kendlike, stats.KendallLike)

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
      print('test-sets:', ' '.join(meta_info.DATA))
    elif FLAGS.language_pair is None:
      print(f'language pairs for {FLAGS.test_set}:',
            ' '.join(meta_info.DATA[FLAGS.test_set]))
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

  if FLAGS.matrix:
    PrintMatrix()
    return

  evs = data.EvalSet(
      FLAGS.test_set, FLAGS.language_pair,
      read_stored_metric_scores=FLAGS.scores)

  if FLAGS.scores:
    PrintScores(evs)
    return

  if FLAGS.echo is not None or FLAGS.echosys is not None:
    flag_val = FLAGS.echo or FLAGS.echosys
    texts = []
    for col in flag_val.split(','):
      if col == 'src':
        texts.append(evs.src)
      elif col == 'doc':
        texts.append(evs.DocsPerSeg())
      elif col == 'domain':
        texts.append(evs.DomainsPerSeg())
      elif col == 'ref':
        texts.append(evs.all_refs[evs.std_ref])
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
