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


def ReadRatingFile(filename) -> dict[str, list[Rating | None]]:
  """Read a file containing sysname/rating entries."""
  ratings = collections.defaultdict(list)  # sys -> [ratings]
  with open(filename) as f:
    for line in f:
      sysname, rating = line.strip().split(maxsplit=1)
      if rating == 'None':
        ratings[sysname].append(None)
      else:
        ratings[sysname].append(Rating.FromDict(json.loads(rating)))
  return ratings


def WriteRatingFile(ratings: dict[str, list[Rating | None]], filename):
  """Write a file containing sysname/rating entries."""
  with open(filename, 'w') as f:
    for sysname, rating_list in ratings.items():
      for rating in rating_list:
        if rating is None:
          f.write(f'{sysname}\tNone\n')
        else:
          f.write(f'{sysname}\t{json.dumps(rating.ToDict())}\n')
