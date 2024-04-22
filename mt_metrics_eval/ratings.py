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
"""Ratings consisting of sub-segment-level error spans."""

import collections
import dataclasses
import json
from typing import Any
import dacite
import glob


@dataclasses.dataclass
class Error:
  """A representation of an error span.

  Attributes:
    start: The starting character offset of the span in the original text.
    end: The end+1 character offset of the span in the original text.
    category: The category.
    severity: The severity.
    score: The original score assigned to this error by a rater or a model.
    is_source_error: True if the span is in the source text.
  """

  start: int
  end: int
  category: str | None = None
  severity: str | None = None
  score: float | None = None
  is_source_error: bool = False

  def ToDict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

  @classmethod
  def FromDict(cls, d: dict[str, Any]) -> 'Error':
    return dacite.from_dict(data_class=Error, data=d)


@dataclasses.dataclass
class Rating:
  """The errors assigned to a translation by a single rater/method."""

  errors: list[Error]

  def ToDict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

  @classmethod
  def FromDict(cls, d: dict[str, Any]) -> 'Rating':
    return dacite.from_dict(data_class=Rating, data=d)


def ReadRatingFile(
    filename: str, default_rater: str
) -> tuple[dict[str, list[Rating | None]], dict[str, list[str]]]:
  """Read a file containing sysname/rating entries."""
  ratings = collections.defaultdict(list)  # sys -> [ratings]
  rater_ids = collections.defaultdict(list)  # sys -> [rater_id]
  with open(filename) as f:
    for line in f:
      cols = line.strip().split('\t')
      if len(cols) == 2:
        sysname, rating = cols
        rater = default_rater
      elif len(cols) == 3:
        sysname, rating, rater = cols
      else:
        raise ValueError(
            f'Expected 2 or 3 columns in rating file. Found {len(cols)}. Line:'
            f' {line}'
        )
      if rating == 'None':
        ratings[sysname].append(None)
        rater_ids[sysname].append(None)
      else:
        ratings[sysname].append(Rating.FromDict(json.loads(rating)))
        rater_ids[sysname].append(rater)
  return ratings, rater_ids


def WriteRatingFile(
    ratings: dict[str, list[Rating | None]],
    filename: str,
    rater_ids_dict: dict[str, list[str | None]],
):
  """Write a file containing sysname/rating entries."""
  with open(filename, 'w') as f:
    for sysname, rating_list in sorted(ratings.items()):
      rater_ids = rater_ids_dict[sysname]
      for rating, rater_id in zip(rating_list, rater_ids):
        if rating is None:
          f.write(f'{sysname}\tNone\n')
        else:
          f.write(f'{sysname}\t{json.dumps(rating.ToDict())}\t{rater_id}\n')
