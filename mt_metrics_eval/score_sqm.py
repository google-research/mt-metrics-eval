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
"""Produce SQM scores from SQM ratings tsv file.

Use with --fields 'system doc doc_id seg_id rater source target score' for cSQM
files, which currently lack field headers.
"""

import collections
import csv
from absl import app
from absl import flags
import glob

flags.DEFINE_string('input', '/dev/stdin', 'Input SQM ratings tsv file.')
flags.DEFINE_string('output', '/dev/stdout', 'Output SQM score file.')
flags.DEFINE_string('level', 'seg', 'Level for output scores: seg, doc, or sys')
flags.DEFINE_string(
    'fields', None,
    'List of fields, must include: system, doc, doc_id seg_id rater score')
flags.DEFINE_bool(
    'raters', False,
    'Write individual rater scores and ids after main score for seg level.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  fields = FLAGS.fields.split() if FLAGS.fields else None

  scores = {}  # sys -> seg -> rater -> score
  docs = {}  # doc -> [beg, end]
  num_segs = 0
  with open(FLAGS.input) as f:
    for row in csv.DictReader(
        f, delimiter='\t', quoting=csv.QUOTE_NONE, fieldnames=fields):
      system, seg_id, rater = row['system'], int(row['seg_id']), row['rater']
      num_segs = max(seg_id, num_segs)
      if system not in scores:
        scores[system] = {}
      if seg_id not in scores[system]:
        scores[system][seg_id] = collections.defaultdict(float)
      scores[system][seg_id][rater] += float(row['score'])

      doc = row['doc']
      if doc not in docs:
        docs[doc] = [9999999, 0]
      b, e = docs[doc]
      docs[doc] = [min(b, seg_id), max(e, seg_id + 1)]

  with open(FLAGS.output, 'w') as f:
    if FLAGS.level == 'seg':
      for system, segs in scores.items():
        for seg_id in range(1, num_segs + 1):
          raters, scores = segs[seg_id].keys(), segs[seg_id].values()
          avg = sum(scores) / len(scores)
          out_str = '%s %f' % (system, avg)
          if FLAGS.raters:
            out_str += ' ' + ' '.join(str(s) for s in scores)
            out_str += ' ' + ' '.join(raters)
          f.write(out_str + '\n')
    elif FLAGS.level == 'doc':
      for system, segs in scores.items():
        for b, e in docs.values():
          score = 0
          for seg_id in range(b, e):
            vals = segs[seg_id].values()
            score += sum(vals) / len(vals)
          f.write('%s %f\n' % (system, score / (e - b)))
    elif FLAGS.level == 'sys':
      for system, segs in scores.items():
        score = 0
        for seg_id in range(1, num_segs + 1):
          vals = segs[seg_id].values()
          score += sum(vals) / len(vals)
        f.write('%s %f\n' % (system, score / num_segs))


if __name__ == '__main__':
  app.run(main)
