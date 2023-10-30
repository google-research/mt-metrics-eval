# MT Metrics Eval V2

MTME is a simple toolkit to evaluate the performance of Machine Translation
metrics on standard test sets from the
[WMT Metrics Shared Tasks](https://statmt.org/wmt22/metrics/index.html).
It bundles data relevant to metric development and evaluation for a
given test set and language pair, and lets you do the following:

- Access source, reference, and MT output text, along with associated
meta-info, for the WMT metrics tasks from 2019-2023. This can be done via
software, or by directly accessing the files in a linux directory
structure, in a straightforward format.
- Access human and automatic metric scores for the above data.
- Reproduce the official results from the WMT metrics tasks. For
WMT22, there is a colab to do this; other years require a bit more work.
- Compute various correlations and perform significance tests on correlation
differences between two metrics.

These can be done on the command line using a python script, or from an
API.

## Installation

You need python 3.9 or later. To install:

```bash
git clone https://github.com/google-research/mt-metrics-eval.git
cd mt-metrics-eval
pip install .
```

## Downloading the data

This must be done before using the toolkit. You can either use the mtme script:

```bash
alias mtme='python3 -m mt_metrics_eval.mtme'
mtme --download  # Puts ~2G of data into $HOME/.mt-metrics-eval.
```

Or download directly, if you're only interested in the data:

```bash
mkdir $HOME/.mt-metrics-eval
cd $HOME/.mt-metrics-eval
wget https://storage.googleapis.com/mt-metrics-eval/mt-metrics-eval-v2.tgz
tar xfz mt-metrics-eval-v2.tgz
```

Once data is downloaded, you can optionally test the install:

```bash
python3 -m unittest mt_metrics_eval.stats_test
python3 -m unittest mt_metrics_eval.data_test  # Takes about 30 seconds.
python3 -m unittest mt_metrics_eval.tasks_test  # Takes about 30 seconds.
```


## Running from the command line

Here are some examples of things you can do with the mtme script. They assume
that the mtme alias above has been set up.

Get information about test sets:

```bash
mtme --list  # List available test sets.
mtme --list -t wmt22  # List language pairs for wmt22.
mtme --list -t wmt22 -l en-de  # List details for wmt22 en-de.
```

Get contents of test sets. Paste doc-id, source, standard reference,
alternative reference to stdout:

```bash
mtme -t wmt22 -l en-de --echo doc,src,refA,refB
```

Outputs from all systems, sequentially, pasted with doc-ids, source, and
reference:

```bash
mtme -t wmt22 -l en-de --echosys doc,src,refA
```

Human and metric scores for all systems, at all granularities:

```bash
mtme -t wmt22 -l en-de --scores > wmt22.en-de.tsv
```

Evaluate metric score files containing tab-separated `system-name score`
entries. For system-level correlations, supply one score per system. For
document-level or segment-level correlations, supply one score per document or
segment, grouped by system, in the same order as text generated using `--echo`
(the same order as the WMT test-set file). Granularity is determined
automatically. Domain-level scores are currently not supported by
this command.

```bash
examples=$HOME/.mt-metrics-eval/mt-metrics-eval-v2/wmt22/metric-scores/en-de

mtme -t wmt22 -l en-de < $examples/BLEU-refA.sys.score
mtme -t wmt22 -l en-de < $examples/BLEU-refA.seg.score
```

Compare to WMT appraise gold scores instead of MQM gold scores:

```bash
mtme -t wmt22 -l en-de -g wmt-appraise < $examples/BLEU-refA.sys.score
mtme -t wmt22 -l en-de -g wmt-appraise < $examples/BLEU-refA.seg.score
```

Compute correlations for two metrics files, and perform tests to determine
whether they are significantly different:

```bash
mtme -t wmt22 -l en-de -i $examples/BLEU-refA.sys.score -c $examples/COMET-22-refA.sys.score
```

Compare all known metrics under specified conditions. This corresponds to one of
the "tasks" in the WMT22 metrics evaluation. The first output line contains all
relevant parameter settings, and subsequent lines show metrics in descending
order of performance, followed by the rank of their significance cluster, the
value of the selected correlation statistic, and a vector of flags to indicate
significant differences with lower-ranked metrics. These examples use k_block=5
for demo purposes; using k_block=100 will approximately match official results
but can take minutes to hours to complete, depending on the task.

```bash
# System-level Pearson
mtme -t wmt22 -l en-de --matrix --k_block 5

# System-level paired-rank accuracy, pooling results across all MQM languages
mtme -t wmt22 -l en-de,zh-en,en-ru --matrix \
  --matrix_corr accuracy --k_block 5

# Segment-level item-wise averaged Kendall-Tau-Acc23 with optimal tie threshold
# using sampling rate of 1.0 (disabling significance testing for demo).
mtme -t wmt22 -l en-de --matrix --matrix_level seg --avg item \
  --matrix_corr KendallWithTiesOpt --matrix_perm_test pairs \
  --matrix_corr_args "{'variant':'acc23', 'sample_rate':1.0}" --k 0
```

## API and Colabs

The colab notebook `mt_metrics_eval.ipynb` contains examples that show how to
use the API to load and summarize data, and compare stored metrics (ones that
participated in the metrics shared tasks) using different criteria. It also
demonstrates how you can incorporate new metrics into these comparisons.

The notebook `wmt22_metrics.ipynb` documents how the official results for the
WMT22 task were generated. We will try to provide similar notebooks for future
evaluations.

The notebook `ties_matter.ipynb` contains the code to reproduce the results
from [Ties Matter: Meta-Evaluating Modern Metrics with Pairwise Accuracy and Tie Calibration](https://arxiv.org/abs/2305.14324).
It also contains examples for how to calculate the proposed pairwise accuracy
with tie calibration.

## Scoring scripts

The scripts `score_mqm` and `score_sqm` can be used to convert MQM and SQM
annotations from [Google's MQM annotation data](
https://github.com/google/wmt-mqm-human-evaluation) into score files in
mt-metrics-eval format. For example:

```bash
git clone https://github.com/google/wmt-mqm-human-evaluation
python3 -m mt_metrics_eval.score_mqm \
  --weights "major:5 minor:1 No-error:0 minor/Fluency/Punctuation:0.1" \
  < wmt-mqm-human-evaluation/generalMT2022/ende/mqm_generalMT2022_ende.tsv \
  > mqm.ende.seg.score
```
This produces an intermediate form with single scores per segment that match
the scores in MTME; the file contains extra columns with rater id and
other info.

Other options let you explore different error weightings or extract scores from
individual annotators.

## File organization and naming convention

### Overview

There is one top-level directory for each test set (e.g. `wmt22`).
Each top-level directory contains the following sub-directories (whose contents
should be obvious from their names):
`documents`, `human-scores`, `metric-scores`, `references`, `sources`, and
`system-outputs`.

In general, a test-set contains data from many language pairs. Each combination
of test-set and language pair (eg wmt22 + de-en) is called an **EvalSet**. This
is the main unit of computation in the toolkit. Each EvalSet consists of a
source text (divided into one or more documents, optionally with domain
membership), reference translations, system outputs to be scored, human gold
scores, and metric scores.

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
- document meta-info:
  - filename: `documents/SRC-TGT.docs`
  - per-line contents: DOMAIN DOCNAME
      - lines match those in the source file
      - documents are assumed to be contiguous blocks of segments
      - DOMAIN tags can be repurposed for categories other than domain, but
        each document must belong to only one category
- references:
  - filename: `references/SRC-TGT.NAME.txt`
      - NAME is the name of this reference, e.g. `refb`. Names cannot be the
     reserved strings `all` or `src`, or contain `.` or `-` characters.
  - per-line contents: text segment
      - lines match those in the source file
- system outputs:
  - filename: `system-outputs/SRC-TGT/NAME.txt`
      - NAME is the name of an MT system or reference
  - per-line contents: text segment
      - lines match those in the source file
- human scores:
  - filename: `human-scores/SRC-TGT.NAME.LEVEL.score`
      - NAME describes the scoring method, e.g. `mqm` or `wmt-z`.
      - LEVEL indicates the granularity of the scores, one of `sys`, `domain`,
        `doc`, or `seg`.
  - per-line contents: [DOMAIN] SYSNAME SCORE
      - DOMAIN is present only if granularity is `domain`
      - SYSNAME must match a NAME in system outputs
      - SCORE may be `None` to indicate a missing score
      - System-level (`sys`) files contain exactly one score per system.
      - Domain-level (`domain`) files contain one score per domain and system.
      - Document-level (`doc`) files contain a block of scores for each system.
      Each block contains the scores for successive documents, in the same order
      they occur in the document info file.
      - Segment-level (`seg`) files contain a block of scores for each system.
      Each block contains the scores for all segments in the system output file,
      in order.
- metric scores:
  - filename `metric-scores/SRC-TGT/NAME-REF.LEVEL.score`
      - NAME is the metric’s base name.
      - REF describes the reference(s) used for this version of the metric,
      either:
          - A list of one or more names separated by `.`, eg `refa` or
          `refa.refb`.
          - The special string `src` to indicate that no reference was used.
          - The special string `all` to indicate that all references were used.
      - LEVEL indicates the granularity of the scores, one of `sys`, `domain`,
        `doc`, or `seg`.
  - per-line contents: [DOMAIN] SYSNAME SCORE
      - Format is identical to human scores, except that `None` entries aren't
      permitted.

## Credits

Inspired by and loosely modeled on
[SacreBLEU](https://github.com/mjpost/sacrebleu).

This is not an official Google product.
