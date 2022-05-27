# MT Metrics Eval V2

This is a simple toolkit to evaluate the performance of Machine Translation
metrics on standard test sets from the
[WMT Metrics Shared Tasks](http://www.statmt.org/wmt20/metrics-task.html).
It bundles all data relevant to metric development and evaluation for a
given test set and language pair, and lets you do the following:

-   List the names of documents and MT systems in the set.
-   Get source and reference texts, and MT system outputs.
-   Get stored human and metric scores for MT system outputs.
-   Compute correlations between human and metric scores, at system, document,
    and segment granularities.
-   Perform significance tests on correlations.

These can be done on the command line using a python script, or from an
API.

## Installation

You need python3. To install:

```bash
git clone https://github.com/google-research/mt-metrics-eval.git
cd mt-metrics-eval
pip install .
```

## Running from the command line

Start by downloading the database (this is also required before using the API):

```bash
alias mtme='python3 -m mt-metrics-eval.mtme'
mtme --download  # Puts ~1G of data into $HOME/.mt-metrics-eval.
mtme --help  # List options.
```

Optionally test the install:

```bash
python3 -m unittest mt_metrics_eval.data_test  # Takes about 30 seconds.
python3 -m unittest mt_metrics_eval.stats_test
```

Get information about test sets:

```bash
mtme --list  # List available test sets.
mtme --list -t wmt20  # List language pairs for wmt20.
mtme --list -t wmt20 -l en-de  # List details for wmt20 en-de.
```

Get contents of test sets. Paste doc-id, source, standard reference,
alternative reference to stdout:

```bash
mtme -t wmt20 -l en-de --echo doc,src,ref,refb
```

Outputs from all systems, sequentially, pasted with doc-ids, source, and
reference:

```bash
mtme -t wmt20 -l en-de --echosys doc,src,ref
```

Human and metric scores for all systems, at all granularities:

```bash
mtme -t wmt20 -l en-de --scores > wmt20.en-de.tsv
```

Evaluate metric score files containing tab-separated "system-name score"
entries. For system-level correlations, supply one score per system. For
document-level or segment-level correlations, supply one score per document or
segment, grouped by system, in the same order as text generated using `--echo`
(the same order as the WMT test-set file). Granularity is determined
automatically:

```bash
examples=$HOME/.mt-metrics-eval/mt-metrics-eval/wmt20/metric-scores/en-de

mtme -t wmt20 -l en-de < $examples/sentBLEU-ref.sys.score
mtme -t wmt20 -l en-de < $examples/sentBLEU-ref.doc.score
mtme -t wmt20 -l en-de < $examples/sentBLEU-ref.seg.score
```

Compare to MQM gold scores instead of WMT gold scores:

```bash
mtme -t wmt20 -l en-de -g mqm < $examples/sentBLEU-ref.sys.score
```

Generate correlations for two metrics files, and perform tests to determine
whether they are significantly different:

```bash
mtme -t wmt20 -l en-de -i $examples/sentBLEU-ref.sys.score -c $examples/COMET-ref.sys.score
```

## Scoring scripts

The scripts `score_mqm` and `score_sqm` can be used to convert MQM and SQM
annotations from [Google's human annotation data for WMT20](
https://github.com/google/wmt-mqm-human-evaluation) into score files in
mt-metrics-eval format. For example:

```bash
wget https://github.com/google/wmt-mqm-human-evaluation/raw/main/ende/mqm_newstest2020_ende.tsv 
python3 -m mt_metrics_eval.score_mqm < mqm_newstest2020_ende.tsv > mqm.seg.score
diff -s mqm.seg.score .mt-metrics-eval/mt-metrics-eval/wmt20/human-scores/en-de.mqm.seg.score
# Files should be identical.
```

Other options let you explore different error weightings or extract scores from
individual annotators.

## API Examples

The colab notebook `mt_metrics_eval.ipynb` contains an example that shows how to
compute correlations and a significance matrix for a given test set and
language pair. This works with stored metric values. The examples below can
also be run from the colab.

WMT correlations for a new metric, all granularities:

```python
from mt_metrics_eval import data

def MyMetric(out, ref):
  """Return a scalar score for given output/reference texts."""
  return -sum(abs(len(a) - len(b)) for a, b in zip(out, ref))

evs = data.EvalSet('wmt20', 'en-de')
scores = {level: {} for level in ['sys', 'doc', 'seg']}
ref = evs.all_refs[evs.std_ref]
for s, out in evs.sys_outputs.items():
  scores['sys'][s] = [MyMetric(out, ref)]
  scores['doc'][s] = [MyMetric(out[b:e], ref[b:e]) for b, e in evs.docs.values()]
  scores['seg'][s] = [MyMetric([o], [r]) for o, r in zip(out, ref)]

# Official WMT correlations.
for level in 'sys', 'doc', 'seg':
  gold_scores = evs.Scores(level, evs.StdHumanScoreName(level))
  sys_names = set(gold_scores) - evs.human_sys_names
  corr = evs.Correlation(gold_scores, scores[level], sys_names)
  print(f'{level}: Pearson={corr.Pearson()[0]:f}, '
        f'Kendall-like={corr.KendallLike()[0]:f}')
```

Correlations using alternative gold scores:

```python
from mt_metrics_eval import data

def MyMetric(out, ref):
  """Return a scalar score for given output/reference texts."""
  return -sum(abs(len(a) - len(b)) for a, b in zip(out, ref))

evs = data.EvalSet('wmt20', 'en-de')
ref = evs.all_refs[evs.std_ref]
scores = {s: [MyMetric(out, ref)] for s, out in evs.sys_outputs.items()}

mqm_scores = evs.Scores('sys', 'mqm')
psqm_scores = evs.Scores('sys', 'psqm')
wmt_scores = evs.Scores('sys', 'wmt-z')
sys_names = set(mqm_scores) - {evs.std_ref}

print(f'MQM Pearson: {evs.Correlation(mqm_scores, scores, sys_names).Pearson()[0]:f}')
print(f'pSQM Pearson: {evs.Correlation(psqm_scores, scores, sys_names).Pearson()[0]:f}')
print(f'WMT Pearson: {evs.Correlation(wmt_scores, scores, sys_names=sys_names).Pearson()[0]:f}')
```

New correlations for a stored metric:

```python
from mt_metrics_eval import data

# Eval set will load more slowly, since we're reading in all stored metrics.
evs = data.EvalSet('wmt20', 'en-de', read_stored_metric_scores=True)
scores = evs.Scores('sys', scorer='BLEURT-extended-ref')

mqm_scores = evs.Scores('sys', 'mqm')
wmt_scores = evs.Scores('sys', 'wmt-z')

qm_sys = set(mqm_scores) - evs.human_sys_names
qm_sys_bp = qm_sys | {'refb', 'refp'}

wmt = evs.Correlation(wmt_scores, scores, qm_sys)
mqm = evs.Correlation(mqm_scores, scores, qm_sys)
wmt_bp = evs.Correlation(wmt_scores, scores, qm_sys_bp)
mqm_bp = evs.Correlation(mqm_scores, scores, qm_sys_bp)

print(f'BLEURT-ext WMT Pearson for qm systems: {wmt.Pearson()[0]:f}')
print(f'BLEURT-ext MQM Pearson for qm systems: {mqm.Pearson()[0]:f}')
print(f'BLEURT-ext WMT Pearson for qm systems plus human: {wmt_bp.Pearson()[0]:f}')
print(f'BLEURT-ext MQM Pearson for qm systems plus human: {mqm_bp.Pearson()[0]:f}')
```

## File organization and naming convention

### Overview

There is one top-level directory for each test set (e.g. wmt20), which may
include multiple language pairs. Each combination of domain and language pair
(e.g. wmt20/de-en) is called an **EvalSet**. This is the main unit of
computation in the toolkit.

Each EvalSet contains a source text (divided into one or more documents),
reference translations, system outputs to be scored, human gold scores, and
metric scores.

Meta information is encoded into directory and file names as specified below.
The convention is intended to be straightforward, but there are a few
subtleties:

- Reference translations can be scored as system outputs. When this is the case,
**the reference files should be copied into the system-outputs directory with
matching names**. For example:
```
      references/de-en.refb.txt → system-outputs/de-en/refb.txt
```
- Metrics can come in different variants according to which reference(s) they
used. This information is encoded into their filenames. To facilitate parsing,
reference names can't contain dashes or dots, as outlined below.
- Metric files must contain scores for all files in the system output directory,
except those that were used as references.
- Human score files don’t have to contain entries for all systems, or even for
all segments for a given system. Missing entries are marked with ‘None’ strings.

### Specification

The filename format and content specification for each kind of file are
described below. Paths are relative to the top-level directory corresponding to
a test set, e.g. wmt20. SRC and TGT designate abbreviations for the
source and target language, e.g. ‘en’. Blanks designate any amount of
whitespace.

- source text:
  - filename: `sources/SRC-TGT.txt`
  - per-line contents: text segment
- document info:
  - filename: `documents/SRC-TGT.docs`
  - per-line contents: TAG DOCNAME
      - lines match those in the source file
      - documents are assumed to be contiguous blocks of segments
      - TAG is currently ignored
- references:
  - filename: `references/SRC-TGT.NAME.txt`
      - NAME is the name of this reference, e.g. ‘refb’. Names cannot be the
     reserved strings ‘all’ or ‘src’, or contain ‘.’ or ‘-’ characters.
  - per-line contents: text segment
      - lines match those in the source file
- system outputs:
  - filename: `system-outputs/SRC-TGT/NAME.txt`
      - NAME is the name of an MT system or reference
  - per-line contents: text segment
      - lines match those in the source file
- human scores:
  - filename: `human-scores/SRC-TGT.NAME.LEVEL.score`
      - NAME describes the scoring method, e.g. ‘mqm’ or ‘wmt-z’.
      - LEVEL indicates the granularity of the scores, one of ‘sys’, ‘doc’, or
      ‘seg’.
  - per-line contents: SYSNAME SCORE
      - SYSNAME must match a NAME in system outputs
      - SCORE may be ‘None’ to indicate a missing score
      - System-level (‘sys’) files contain exactly one score per system.
      - Document-level (‘doc’) files contain a block of scores for each system.
      Each block contains the scores for successive documents, in the same order
      they occur in the document info file.
      - Segment-level (‘seg’) files contain a block of scores for each system.
      Each block contains the scores for all segments in the system output file,
      in order.
- metric scores:
  - filename `metric-scores/SRC-TGT/NAME-REF.LEVEL.score`
      - NAME is the metric’s base name.
      - REF describes the reference(s) used for this version of the metric,
      either:
          - A list of one or more names separated by ‘.’, eg ‘refa’ or
          ‘refa.refb’.
          - The special string ‘src’ to indicate that no reference was used.
          - The special string ‘all’ to indicate that all references were used.
      - LEVEL indicates the granularity of the scores, one of sys, doc, or seg.
  - per-line contents: SYSNAME SCORE
      - Format is identical to human scores, except that ‘None’ entries aren’t 
      permitted.

## Credits

Inspired by and loosely modeled on
[SacreBLEU](https://github.com/mjpost/sacrebleu).

This is not an official Google product.
