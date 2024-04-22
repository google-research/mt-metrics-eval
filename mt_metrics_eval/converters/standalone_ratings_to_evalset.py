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
"""Convert standalone ratings file to EvalSet format files."""

import os
from absl import app
from absl import flags
from mt_metrics_eval import data
from mt_metrics_eval import ratings
from mt_metrics_eval import standalone_ratings
import glob

flags.DEFINE_multi_string(
    'ratings_file', None,
    'Ratings jsonl file to be converted.', required=True)
flags.DEFINE_string(
    'test_set', None, 'Test set, eg wmt20.', required=True)
flags.DEFINE_string(
    'language_pair', None, 'Language pair, eg en-de.', required=True)
flags.DEFINE_string(
    'output_dir', None,
    'Directory in which to write output files.', required=True)
flags.DEFINE_string(
    'prefix', '',
    'Prefix for output files. Full name is {prefix}{rater}.seg.rating.')
flags.DEFINE_bool(
    'anonymize_raters', False, 'Anonymize rater names.')
flags.DEFINE_bool(
    'merge_raters', False,
    'By default, conversion produces a separate rating file for each rater, '
    'even when raters annotate disjoint sets of items. This option will write '
    'only a single file {prefix}merged.seg.rating by merging contributions '
    'when possible, ie when rater contributions are disjoint; otherwise it '
    'will write separate files. Note that it is not possible to recover '
    'original rater names with this option.')
flags.DEFINE_bool(
    'strict', True, 'Ensure text-level matches with the EvalSet.')
flags.DEFINE_string(
    'rater_key_file', None,
    'Write rater rename key to this file, with entries of the form '
    'old-name\tnew-name. New names are identical to old names unless '
    'anonymize_raters is True or the original rater names are None.')
flags.DEFINE_string(
    'echo_ratings_file', None,
    'Write ratings in standalone format to this file. These many not be '
    'identical to the orginal entries due to dropping some fields and changing '
    'field order.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  ratings_list = []
  for ratings_file in FLAGS.ratings_file:
    ratings_list.extend(standalone_ratings.ReadRatingFile(ratings_file))
  evs = data.EvalSet(FLAGS.test_set, FLAGS.language_pair)
  ratings_dict, raters_key, rater_ids_dict = (
      standalone_ratings.RatingsListToEvalSetRatings(
          ratings_list, evs, FLAGS.anonymize_raters, FLAGS.strict
      )
  )

  if FLAGS.echo_ratings_file:
    standalone_ratings.WriteRatingFile(ratings_list, FLAGS.echo_ratings_file)

  if FLAGS.merge_raters:
    new_ratings, new_rater_ids = standalone_ratings.MergeEvalSetRaters(
        ratings_dict, evs, rater_ids_dict
    )
    if new_ratings:
      ratings_dict = {'merged': new_ratings}
      rater_ids_dict = {'merged': new_rater_ids}

  for rater, evs_ratings in ratings_dict.items():
    filename = os.path.join(
        FLAGS.output_dir, f'{FLAGS.prefix}{rater}.seg.rating')
    ratings.WriteRatingFile(evs_ratings, filename, rater_ids_dict[rater])

  if FLAGS.rater_key_file:
    with open(FLAGS.rater_key_file, 'w') as f:
      for rater, new_rater in raters_key.items():
        f.write(f'{rater}\t{new_rater}\n')


if __name__ == '__main__':
  app.run(main)
