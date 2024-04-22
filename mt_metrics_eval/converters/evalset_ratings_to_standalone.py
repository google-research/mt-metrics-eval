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

flags.DEFINE_list(
    'evalset_ratings_files', None,
    'Comma-separated list of evalset-format ratings files to read. Filenames '
    'are assumed to be in the form {language_pair}.{name}.seg.rating, where '
    'name is of the form {prefix}.{rater}.',
    required=True)
flags.DEFINE_string(
    'ratings_file', None,
    'Standalone ratings jsonl file to write.', required=True)
flags.DEFINE_string(
    'test_set', None, 'Test set, eg wmt20.', required=True)
flags.DEFINE_string(
    'language_pair', None, 'Language pair, eg en-de.', required=True)
flags.DEFINE_string(
    'rater_key_file', None,
    'Use a rater_key_file previously written by standalone_ratings_to_evalset '
    'to deanonymize raters in the output file.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  raters_key = {}
  if FLAGS.rater_key_file:
    with open(FLAGS.rater_key_file, 'r') as f:
      for line in f:
        k, v = line.rstrip().split('\t')
        raters_key[k] = v
    # reverse the mapping
    raters_key = {v: k for k, v in raters_key.items()}

  evs = data.EvalSet(FLAGS.test_set, FLAGS.language_pair)

  evalset_ratings: dict[str, dict[str, list[ratings.Rating | None]]] = {}
  evalset_rater_ids: dict[str, dict[str, list[str | None]]] = {}
  for filename in FLAGS.evalset_ratings_files:
    if not filename: continue
    _, name, _ = evs.ParseHumanScoreFilename(
        os.path.basename(filename), rating_file=True)
    rating_name = name.rsplit('.', maxsplit=1)[-1]
    evalset_ratings[rating_name], evalset_rater_ids[rating_name] = (
        ratings.ReadRatingFile(filename, rating_name)
    )

  ratings_list = standalone_ratings.EvalSetRatingsToRatingsList(
      evalset_ratings, evs, evalset_rater_ids, raters_key)

  standalone_ratings.WriteRatingFile(ratings_list, FLAGS.ratings_file)


if __name__ == '__main__':
  app.run(main)
