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
"""Produce MQM scores from MQM ratings tsv file."""

import collections
import csv
import json
from absl import app
from absl import flags
import glob

flags.DEFINE_string('input', '/dev/stdin', 'Input MQM ratings tsv file.')
flags.DEFINE_string('output', '/dev/stdout', 'Output MQM score file.')
flags.DEFINE_string(
    'weights', 'Major:5 Minor:1 Neutral:0 '
    'Major/Non-translation!:25 Minor/Fluency/Punctuation:0.1',
    'List of weight specs, in format: "severity[/category[/subcategory]]:wt". '
    'The most specific match is applied to each error.')
flags.DEFINE_string(
    'weights_sep', ' ', 'Separator character between items in weights lists.')
flags.DEFINE_bool('unbabel', False, 'Input tsv is in Unbabel format.')
flags.DEFINE_bool(
    'recompute_unbabel', False,
    'Apply Google-style weights to Unbabel ratings rather than reading scores '
    'directly from mqm field in last column of tsv.')
flags.DEFINE_bool(
    'force_contiguous', True,
    'Raise an error if annotated segments within a doc aren\'t contiguous')
flags.DEFINE_string(
    'doc_id', 'doc_id',
    'Name of field containing 1-based id of segment within document')

FLAGS = flags.FLAGS


def Score(weights, items):
  items = [x.lower() for x in items]
  while items:
    if '/'.join(items) in weights:
      return weights['/'.join(items)]
    items = items[:-1]
  return 0


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  weights = {}
  for e in FLAGS.weights.split(FLAGS.weights_sep):
    c, w = e.split(':')
    weights[c.lower()] = float(w)

  scores = {}  # sys -> doc > doc_id -> rater -> [score]
  quoting = csv.QUOTE_MINIMAL if FLAGS.unbabel else csv.QUOTE_NONE
  with open(FLAGS.input) as f:
    for row in csv.DictReader(f, delimiter='\t', quoting=quoting):
      system, doc, doc_id = row['system'], row['doc'], int(row[FLAGS.doc_id])
      if FLAGS.unbabel and not FLAGS.recompute_unbabel:
        score = json.loads(row['misc'])['mqm']
      else:
        score = Score(weights, [row['severity']] + row['category'].split('/'))
      if system not in scores:
        scores[system] = {}
      if doc not in scores[system]:
        scores[system][doc] = {}
      if doc_id not in scores[system][doc]:
        scores[system][doc][doc_id] = collections.defaultdict(list)
      scores[system][doc][doc_id][row['rater']].append(score)

  if FLAGS.force_contiguous:
    for system in scores:
      for doc in scores[system]:
        ids = sorted(scores[system][doc])
        if ids != list(range(min(ids), max(ids) + 1)):
          raise ValueError(f'Non-contiguous segments for {system}/{doc}')

  with open(FLAGS.output, 'w') as f:
    for system in scores:
      for doc in scores[system]:
        for doc_id in sorted(scores[system][doc]):
          rater_scores = {}
          for rater, vals in scores[system][doc][doc_id].items():
            if FLAGS.unbabel and not FLAGS.recompute_unbabel:
              rater_scores[rater] = sum(vals) / len(vals)
            else:
              rater_scores[rater] = sum(vals)
          global_score = sum(rater_scores.values()) / len(rater_scores)
          if not FLAGS.unbabel or FLAGS.recompute_unbabel:
            global_score *= -1
          f.write(f'{system}\t{doc}\t{doc_id}\t{global_score}')
          for rater in sorted(rater_scores):
            f.write(f'\t{rater}={rater_scores[rater]}')
          f.write('\n')

if __name__ == '__main__':
  app.run(main)
