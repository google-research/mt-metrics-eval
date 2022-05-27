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
"""Meta-information for standard datasets."""

import dataclasses


@dataclasses.dataclass
class MetaInfo:
  """Meta information for test-sets and language pairs."""
  std_ref: str
  std_gold_sys: str
  std_gold_doc: str
  std_gold_seg: str
  outlier_systems: set[str]
  primary_metrics: set[str]  # Includes any baseline metrics.

WMT19 = MetaInfo('ref', 'wmt-z', '', 'wmt-raw', set(), set())
WMT20 = MetaInfo('ref', 'wmt-z', 'wmt-raw', 'wmt-raw', set(), set())
WMT21_PRIMARIES = {
    'bleurt-20', 'COMET-MQM_2021',
    'COMET-QE-MQM_2021', 'C-SPECpn', 'MEE2',
    'MTEQA', 'OpenKiwi-MQM', 'regEMT', 'YiSi-1',
    'YiSi-2', 'BERTScore', 'sentBLEU', 'BLEU', 'chrF', 'Prism', 'TER'
}
WMT21 = MetaInfo('refA', 'wmt-z', '', 'wmt-raw', set(), WMT21_PRIMARIES)

DATA = {
    'wmt21.news': {
        'en-cs': WMT21,
        'en-de': dataclasses.replace(
            WMT21,
            std_ref='refC',
            std_gold_sys='mqm',
            std_gold_seg='mqm',
            primary_metrics=WMT21_PRIMARIES | {'cushLEPOR(LM)'}),
        'en-ha': WMT21,
        'en-is': WMT21,
        'en-ja': WMT21,
        'en-ru': dataclasses.replace(
            WMT21,
            std_ref='refA',
            std_gold_sys='mqm',
            std_gold_seg='mqm',
            primary_metrics=WMT21_PRIMARIES | {'hLEPOR'}),
        'en-zh': WMT21,
        'cs-en': WMT21,
        'de-en': WMT21,
        'de-fr': WMT21,
        'fr-de': WMT21,
        'ha-en': WMT21,
        'is-en': WMT21,
        'ja-en': WMT21,
        'ru-en': WMT21,
        'zh-en': dataclasses.replace(
            WMT21,
            std_ref='refB',
            std_gold_sys='mqm',
            std_gold_seg='mqm',
            primary_metrics=WMT21_PRIMARIES | {'cushLEPOR(LM)', 'RoBLEURT'}),
    },
    'wmt21.tedtalks': {
        'en-de': dataclasses.replace(
            WMT21,
            std_ref='refA',
            std_gold_sys='mqm',
            std_gold_seg='mqm',
            primary_metrics=WMT21_PRIMARIES | {'cushLEPOR(LM)'}),
        'en-ru': dataclasses.replace(
            WMT21,
            std_ref='refA',
            std_gold_sys='mqm',
            std_gold_seg='mqm',
            primary_metrics=WMT21_PRIMARIES | {'hLEPOR'}),
        'zh-en': dataclasses.replace(
            WMT21,
            std_ref='refB',
            std_gold_sys='mqm',
            std_gold_seg='mqm',
            primary_metrics=WMT21_PRIMARIES | {'cushLEPOR(LM)', 'RoBLEURT'}),
    },
    'wmt21.flores': {
        'bn-hi': WMT21,
        'hi-bn': WMT21,
        'xh-zu': WMT21,
        'zu-xh': WMT21,
    },
    'wmt20': {
        'cs-en': dataclasses.replace(
            WMT20,
            outlier_systems={'zlabs-nlp.1149', 'CUNI-DocTransformer.1457'}),
        'de-en': dataclasses.replace(
            WMT20,
            outlier_systems={'yolo.1052', 'zlabs-nlp.1153',
                             'WMTBiomedBaseline.387'}),
        'en-cs': dataclasses.replace(
            WMT20,
            outlier_systems={'zlabs-nlp.1151', 'Online-G.1555'}),
        'en-de': dataclasses.replace(
            WMT20,
            outlier_systems={'zlabs-nlp.179', 'WMTBiomedBaseline.388',
                             'Online-G.1556'}),
        'en-iu': dataclasses.replace(
            WMT20,
            outlier_systems={'UEDIN.1281', 'OPPO.722', 'UQAM_TanLe.521'}),
        'en-ja': dataclasses.replace(
            WMT20,
            outlier_systems={'Online-G.1557', 'SJTU-NICT.370'}),
        'en-pl': dataclasses.replace(
            WMT20,
            outlier_systems={'Online-Z.1634', 'zlabs-nlp.180',
                             'Online-A.1576'}),
        'en-ru': WMT20,
        'en-ta': dataclasses.replace(
            WMT20,
            outlier_systems={'TALP_UPC.1049', 'SJTU-NICT.386',
                             'Online-G.1561'}),
        'en-zh': WMT20,
        'iu-en': dataclasses.replace(
            WMT20,
            outlier_systems={'NiuTrans.1206', 'Facebook_AI.729'}),
        'ja-en': dataclasses.replace(
            WMT20,
            outlier_systems={'Online-G.1564', 'zlabs-nlp.66', 'Online-Z.1640'}),
        'km-en': WMT20,
        'pl-en': dataclasses.replace(
            WMT20,
            outlier_systems={'zlabs-nlp.1162'}),
        'ps-en': WMT20,
        'ru-en': dataclasses.replace(
            WMT20,
            outlier_systems={'zlabs-nlp.1164'}),
        'ta-en': dataclasses.replace(
            WMT20,
            outlier_systems={'Online-G.1568', 'TALP_UPC.192'}),
        'zh-en': dataclasses.replace(
            WMT20,
            outlier_systems={'WMTBiomedBaseline.183'})
    },
    'wmt19': {
        'de-cs': WMT19,
        'de-en': WMT19,
        'de-fr': WMT19,
        'en-cs': WMT19,
        'en-de': WMT19,
        'en-fi': WMT19,
        'en-gu': WMT19,
        'en-kk': WMT19,
        'en-lt': WMT19,
        'en-ru': WMT19,
        'en-zh': WMT19,
        'fi-en': WMT19,
        'fr-de': WMT19,
        'gu-en': WMT19,
        'kk-en': WMT19,
        'lt-en': WMT19,
        'ru-en': WMT19,
        'zh-en': WMT19,
    }
}

