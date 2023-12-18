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
"""Tests for the ratings module."""

from mt_metrics_eval import ratings
import unittest


class ErrorTest(unittest.TestCase):

  def test_error_serialization(self):
    error = ratings.Error(1, 4, 'cat', None, 5, True)
    serialized = error.ToDict()
    expected = {
        'start': 1,
        'end': 4,
        'category': 'cat',
        'severity': None,
        'score': 5,
        'is_source_error': True,
    }
    self.assertEqual(serialized, expected)
    deserialized = ratings.Error.FromDict(serialized)
    self.assertEqual(deserialized, error)


class RatingTest((unittest.TestCase)):

  def test_rating_serialization(self):
    rating = ratings.Rating(
        [
            ratings.Error(1, 4, 'cat', None, 5, True),
            ratings.Error(0, 1),
        ]
    )
    serialized = rating.ToDict()
    expected = {
        'errors': [
            {
                'start': 1,
                'end': 4,
                'category': 'cat',
                'severity': None,
                'score': 5,
                'is_source_error': True,
            },
            {
                'start': 0,
                'end': 1,
                'category': None,
                'severity': None,
                'score': None,
                'is_source_error': False,
                },
        ]
    }
    self.assertEqual(serialized, expected)
    deserialized = ratings.Rating.FromDict(serialized)
    self.assertEqual(deserialized, rating)


if __name__ == "__main__":
  unittest.main()
