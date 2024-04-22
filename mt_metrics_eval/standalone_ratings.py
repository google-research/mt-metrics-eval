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
"""Self-contained ratings consisting of sub-segment-level error spans."""

import dataclasses
import json
from typing import Any
import dacite
from mt_metrics_eval import data
from mt_metrics_eval import ratings
import glob


@dataclasses.dataclass
class Rating:
  """The errors assigned to a translation by a single rater/method.

  Attributes:
    source: The source text.
    hypothesis: The translation.
    errors: The list of errors.
    document_id: The ID of the document where the source text comes from.
    segment_id: The 0-indexed offset of this segment in the test set.
    system_id: The ID of the system that generated the translation.
    rater_id: The ID of the rater/method that annotated the errors.
    src_lang: The source language code.
    tgt_lang: The target language code.
  """

  source: str
  hypothesis: str
  errors: list[ratings.Error]
  document_id: str | None = None
  segment_id: int | None = None
  system_id: str | None = None
  rater_id: str | None = None
  src_lang: str | None = None
  tgt_lang: str | None = None

  def ToDict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

  @classmethod
  def FromDict(cls, d: dict[str, Any]) -> 'Rating':
    return dacite.from_dict(data_class=Rating, data=d)


def ReadRatingFile(filename) -> list[Rating]:
  """Read a file containing a list of Ratings."""
  ratings_list = []
  with open(filename) as f:
    for line in f:
      ratings_list.append(Rating.FromDict(json.loads(line)))
  return ratings_list


def WriteRatingFile(ratings_list: list[Rating], filename):
  """Write a list of Ratings to file."""
  with open(filename, 'w') as f:
    for rating in ratings_list:
      f.write(f'{json.dumps(rating.ToDict())}\n')


def _RenameRaters(
    ratings_list: list[Rating], anonymize: bool
) -> dict[Any, str]:
  """Rename original rater names."""
  rater_ids = sorted(set(rating.rater_id for rating in ratings_list))
  # If all raters start with 'rater', don't anonymize to avoid a confusing
  # renaming due to sorting rater10 before rater2.
  if anonymize and not all(r.startswith('rater') for r in rater_ids if r):
    return {rater: f'rater{i + 1}' for i, rater in enumerate(rater_ids)}
  else:
    if None in rater_ids and 'rater' in rater_ids:
      raise ValueError(
          'Attempt to rename rater "None" to "rater" failed because "rater"'
          ' already exists.'
      )
    return {
        rater: (rater if rater is not None else 'rater') for rater in rater_ids
    }


def _CheckRating(
    rating: Rating, evs: data.EvalSet, rating_id: int, strict: bool = True
):
  """Check rating for compatibility with evs, with text match if strict."""
  if rating.segment_id is None or rating.system_id is None:
    raise ValueError(
        f'Rating {rating_id}: conversion requires non-null segment and system '
        'ids.'
    )
  rating_id += 1  # 1-based for human consumption
  seg = rating.segment_id
  if seg >= len(evs.src):
    raise ValueError(f'Segment offset is too big in rating {rating_id}: {seg}')
  if rating.document_id is not None:
    if rating.document_id not in evs.doc_names:
      raise ValueError(
          f'Unknown doc in rating {rating_id}: {rating.document_id}')
    doc_beg, doc_end = evs.docs[rating.document_id]
    if seg < doc_beg or seg >= doc_end:
      raise ValueError(
          f'Bad segment offset for doc {rating.document_id} in rating '
          '{rating_id}: {seg}')
  if rating.system_id not in evs.sys_names:
    raise ValueError(f'Unknown sys in rating {rating_id}: {rating.system_id}')
  if rating.src_lang is not None and rating.src_lang != evs.src_lang:
    raise ValueError(
        f'Bad source language in rating {rating_id}: {rating.src_lang}')
  if rating.tgt_lang is not None and rating.tgt_lang != evs.tgt_lang:
    raise ValueError(
        f'Bad target language in rating {rating_id}: {rating.tgt_lang}')
  if strict:
    # We assume that the rating is internally consistent, so if the source and
    # hypothesis match evs, all error spans will be in range.
    if rating.source != evs.src[seg]:
      raise ValueError(f'Source segment mismatch in rating {rating_id}')
    if rating.hypothesis != evs.sys_outputs[rating.system_id][seg]:
      raise ValueError(f'Hypothesis segment mismatch in rating {rating_id}')


def RatingsListToEvalSetRatings(
    ratings_list: list[Rating],
    evs: data.EvalSet,
    anonymize_raters: bool = False,
    strict: bool = True,
) -> tuple[
    dict[str, dict[str, list[ratings.Rating | None]]],
    dict[str, str],
    dict[str, dict[str, list[str | None]]],
]:
  """Convert Ratings list to EvalSet-style ratings dict and rater rename map."""
  new_rater_names = _RenameRaters(ratings_list, anonymize_raters)
  ratings_dict = {}  # rating_name -> {sys: [rating]}
  rater_ids_dict = {}  # rating_name -> {sys: [rater_id]}
  for rating_id, rating in enumerate(ratings_list):
    _CheckRating(rating, evs, rating_id, strict)
    rater = new_rater_names[rating.rater_id]
    if rater not in ratings_dict:
      ratings_dict[rater] = {s: [None] * len(evs.src) for s in evs.sys_names}
      rater_ids_dict[rater] = {s: [None] * len(evs.src) for s in evs.sys_names}
    if ratings_dict[rater][rating.system_id][rating.segment_id] is not None:
      # Nothing in the Rating spec precludes this, but it's probably something
      # we want to enforce.
      raise ValueError(
          f'Rating already exists for system/rater/segment: {rating_id}'
      )
    evs_rating = ratings.Rating(rating.errors)
    ratings_dict[rater][rating.system_id][rating.segment_id] = evs_rating
    rater_ids_dict[rater][rating.system_id][rating.segment_id] = rater
  return ratings_dict, new_rater_names, rater_ids_dict


def MergeEvalSetRaters(
    evs_ratings: dict[str, dict[str, list[ratings.Rating | None]]],
    evs: data.EvalSet,
    evs_rater_ids: dict[str, dict[str, list[str | None]]],
) -> tuple[dict[str, list[ratings.Rating | None]], dict[str, list[str | None]]]:
  """Merge disjoint ratings from multiple raters into single-rater dict."""
  new_ratings = {s: [None] * len(evs.src) for s in evs.sys_names}
  new_rater_ids = {s: [None] * len(evs.src) for s in evs.sys_names}
  for rating_name in evs_ratings:
    for system_id in evs_ratings[rating_name]:
      rater_ids = evs_rater_ids[rating_name][system_id]
      for seg, evs_rating in enumerate(evs_ratings[rating_name][system_id]):
        if evs_rating is not None:
          if new_ratings[system_id][seg] is not None:
            raise ValueError(
                f'Found duplicate rating for system/segment: {system_id}/{seg}'
            )
          new_ratings[system_id][seg] = evs_rating
          new_rater_ids[system_id][seg] = rater_ids[seg]
  return new_ratings, new_rater_ids


def EvalSetRatingsToRatingsList(
    evs_ratings: dict[str, dict[str, list[ratings.Rating | None]]],
    evs: data.EvalSet,
    evs_rater_ids: dict[str, dict[str, list[str | None]]],
    rename_raters: dict[str, str] | None = None,
) -> list[Rating]:
  """Convert an EvalSet-style ratings dict to a list of Ratings."""
  docs_per_seg = evs.DocsPerSeg()
  ratings_list = []
  for rating_name in evs_ratings:
    for system_id in evs_ratings[rating_name]:
      rater_ids = evs_rater_ids[rating_name][system_id]
      for seg, evs_rating in enumerate(evs_ratings[rating_name][system_id]):
        if evs_rating is None:
          continue
        rater = rater_ids[seg]
        rating = Rating(
            source=evs.src[seg],
            hypothesis=evs.sys_outputs[system_id][seg],
            errors=evs_rating.errors,
            document_id=docs_per_seg[seg],
            segment_id=seg,
            system_id=system_id,
            rater_id=rename_raters[rater] if rename_raters else rater,
        )
        ratings_list.append(rating)
  return ratings_list
