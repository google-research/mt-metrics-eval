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
"""Verify human and metric scores files, optionally repair metrics files."""

import os
from absl import app
from absl import flags
from mt_metrics_eval import data
import glob

flags.DEFINE_string(
    'scores_file', None,
    'Scores file to be verified. If not supplied, check all scores files for '
    'given test_set, doing on-the-fly-repair, ie only report non-repairable '
    'errors.')
flags.DEFINE_bool('human_scores', False, 'File contains human scores.')
flags.DEFINE_string(
    'data_dir', None, 'Optional root directory for mt_metrics_eval data.')
flags.DEFINE_string(
    'test_set', None,
    'Name of test_set to which metric pertains.', required=True)
flags.DEFINE_string(
    'language_pair', None,
    'Language pair, must exist for test_set.', required=True)
flags.DEFINE_string(
    'repair', None,
    'Write a repaired version of scores_file to this file. This will be a '
    'verbatim copy if scores_file is correct. No action if --human_scores is '
    'set.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if FLAGS.scores_file:
    scores_file = os.path.basename(FLAGS.scores_file)
  else:
    scores_file = None
  read_all_scores = scores_file is None
  evs = data.EvalSet(FLAGS.test_set, FLAGS.language_pair,
                     read_stored_metric_scores=read_all_scores,
                     path=FLAGS.data_dir,
                     strict=False)

  if read_all_scores:
    return

  # Check filename conventions, fail with error if incorrect.
  if FLAGS.human_scores:
    lp, name, level = evs.ParseHumanScoreFilename(scores_file)
    if lp != FLAGS.language_pair:
      raise ValueError(
          f'Language pair {lp} from scores file doesn\'t match flag.')
  else:
    name, level = evs.ParseMetricFilename(scores_file)
    evs.ParseMetricName(name)

  # Check contents, optionally repair missing-system errors.
  scores_map = data.ReadScoreFile(FLAGS.scores_file)
  added = evs.CheckScores(
      scores_map, name, level, FLAGS.human_scores, FLAGS.repair)
  if added:
    print(f'Added dummy scores (0s) for missing outputs: {added}')

  if FLAGS.repair and not FLAGS.human_scores:
    with open(FLAGS.repair, 'w') as f:
      for sysname, scores in scores_map.items():
        f.write('\n'.join([f'{sysname}\t{s}' for s in scores]) + '\n')


if __name__ == '__main__':
  app.run(main)
