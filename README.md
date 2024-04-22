# MT Metrics Eval V2

MTME is a simple toolkit to evaluate the performance of Machine Translation
metrics on standard test sets such as those from the
[WMT Metrics Shared Tasks](https://wmt-metrics-task.github.io).
It bundles data relevant to metric development and evaluation for a
given test set and language pair, and lets you do the following:

- Access source, reference, and MT output text, along with associated
meta-info, for the WMT metrics tasks from 2019 on. This can be done via
software, or by directly accessing the files in a linux directory
structure, in a straightforward format.
- Access human and automatic metric scores for the above data, and MQM ratings
for some language pairs.
- Reproduce the official results from the WMT metrics tasks. For
WMT22 on, there are colabs to do this; other years require more work.
- Compute various correlations and perform significance tests on correlation
differences between two metrics.

These can be done on the command line using a python script, or from an
API.

## Installation

You need python 3.10 or later. To install:

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
python3 -m unittest discover mt_metrics_eval "*_test.py"  # Takes ~70 seconds.
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

The notebooks `wmt22_metrics.ipynb` and `wmt23_metrics.ipynb` document how the
official results for these tasks were generated.
We will try to provide similar notebooks for future evaluations.

The notebook `ties_matter.ipynb` contains the code to reproduce the results
from [Ties Matter: Meta-Evaluating Modern Metrics with Pairwise Accuracy and Tie Calibration](https://arxiv.org/abs/2305.14324).
It also contains examples for how to calculate the proposed pairwise accuracy
with tie calibration.

## MQM Ratings

MTME also supports representing MQM ratings.
The ratings are stored as `rating.Rating` objects in the `EvalSet`.
They can be accessed via the `EvalSet.Ratings()` function.
`Ratings()` returns a dictionary that maps between the name of a set of
ratings and the ratings themselves, one per segment.
Each entry can either represent:

- An individual rater's ratings, in which the key is the ID of the rater
- A metric's ratings, in which the key is the ID of the system that predicted the rating
- A combined set of ratings that come from different raters, in which the key
is the name for this group of ratings. This could be used if there was a logical
"round" of ratings from different raters, like a full round of ratings collected
as part of a WMT evaluation.

The IDs of the raters who rated the segments can be accessed via
`EvalSet.RaterIdsPerSeg()`. It returns a dict that is parallel to an entry
in `EvalSet.Ratings()` that lists the individual rater IDs for each rating or
`None` if there was no rating.
For an individual rater's ratings or a metric's ratings, these are typically
that rater's ID or the name of the metric. For a combined set of ratings, this
will contain the per-segment rater IDs.

For each year of WMT for which ratings are included in MTME, there is a rating
entry for each individual rater. If there was a logical grouping of ratings,
like a round of ratings that were collected at the same time, those are also
included.
Here are the ratings that are currently available:

| Dataset | Language Pair | Ratings |
| ------- | ------------- | ------- |
| wmt20   | en-de         | <ul><li>"mqm.rater1"-"mqm.rater6": The individual rater's ratings. Each segment was rated up to 3 times, and there is no clear definition of a round of ratings, so no combined set of ratings is included.</li></ul> |
| wmt20   | zh-en         | <ul><li>"mqm.rater1"-"mqm.rater6": The individual rater's ratings. Each segment was rated up to 3 times, and there is no clear definition of a round of ratings, so no combined set of ratings is included.</li></ul> |
| wmt21.news | en-de | <ul><li>"mqm.rater1"-"mqm.rater14": The individual rater's ratings</li><li>"mqm.merged": The combined ratings of rater1-14 that were used in the WMT evaluation</li></ul> |
| wmt21.news | zh-en | <ul><li>"mqm.rater1"-"mqm.rater9": The individual rater's ratings</li><li>"mqm.merged": The combined ratings of rater1-9 that were used in the WMT evaluation</li></ul> |
| wmt21.tedtalks | en-de | <ul><li>"mqm.rater1"-"mqm.rater4": The individual rater's ratings</li><li>"mqm.merged": The combined ratings of rater1-4 that were used in the WMT evaluation</li></ul> |
| wmt21.tedtalks | zh-en | <ul><li>mqm.rater1-mqm.rater9: The individual rater's ratings</li><li>"mqm.merged": The combined ratings of rater1-9 that were used in the WMT evaluation</li></ul> |
| wmt22 | en-de | <ul><li>"mqm.rater1"-"mqm.rater7": The individual rater's ratings (from all rounds; see below)</li><li>"mqm.merged": The combined ratings of rater1-7 that were used in the WMT evaluation</li><li>"round2.mqm.merged": A second round of ratings collected from rater1-7 (these were not part of the WMT evaluation)</li><li>"round3.mqm.merged": A third round of ratings collected from rater1-7 (these were not part of the WMT evaluation)</li></ul> |
| wmt22 | en-ru | <ul><li>"mqm.rater1"-"mqm.rater4": The individual rater's ratings</li><li>"mqm.merged": The combined ratings of rater1-4 that were used in the WMT evaluation</li> |
| wmt22 | zh-en | <ul><li>"mqm.rater1"-"mqm.rater12": The individual rater's ratings</li><li>"mqm.merged": The combined ratings of rater1-12 that were used in the WMT evaluation</li> |
| wmt23 | en-de | <ul><li>"mqm.rater1"-"mqm.rater10": The individual rater's ratings</li><li>"mqm.merged": The combined ratings of rater1-10 that were used in the WMT evaluation</li> |
| wmt23 | zh-en | <ul><li>"mqm.rater1"-"mqm.rater8": The individual rater's ratings. A small subset of segments were rated by all of the raters, so there is no clear definition of a round of ratings, so no merged set of ratings is included.</li> |

Note that the ratings might differ slightly from the ratings that were released
as part of the original WMT evaluations. The released data and the translations
in MTME were sometimes different (e.g., punctuation was introduced or removed,
whitespace inserted, etc.), which made it difficult to map the MQM ratings to
character offsets in the MTME translations.
We wrote scripts to fix the ratings so they would match the MTME versions, but
this was sometimes lossy and not always possible, so some ratings might be
different or even missing.
This is less of a problem with more recent WMT years.


## Conversion scripts

The `converters` module contains scripts to convert between different formats
for ratings and scores.

For example, to convert MQM annotations from [Google's tsv annotation format](
https://github.com/google/wmt-mqm-human-evaluation) into scores:

```bash
git clone https://github.com/google/wmt-mqm-human-evaluation
python3 -m mt_metrics_eval.converters.score_mqm \
  --weights "major:5 minor:1 No-error:0 minor/Fluency/Punctuation:0.1" \
  < wmt-mqm-human-evaluation/generalMT2022/ende/mqm_generalMT2022_ende.tsv \
  > mqm.ende.seg.score
```

To convert MTME-format MQM annotations into standalone json files that bundle
all relevant information:

```bash
python3 -m mt_metrics_eval.converters.evalset_ratings_to_standalone \
  --evalset_ratings_files  $HOME/.mt-metrics-eval/mt-metrics-eval-v2/wmt23/human-scores/en-de.mqm.merged.seg.rating \
  --language_pair en-de \
  --test_set wmt23 \
  --ratings_file en-de.mqm.standalone.jsonl
```

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
- human MQM ratings:
  - filename: `human-scores/SRC-TGT.RATING_NAME.seg.rating`
      - RATING_NAME describes the name for the collection of ratings. This can be the name of an individual rater or a name like "mqm.merged", which means multiple rater's ratings have been merged into a single collection of ratings.
  - per-line contents: SYSNAME RATING [RATER_ID]
      - SYSNAME must match a NAME in system outputs
      - A JSON-serialized `ratings.Rating` object or "None" if there is no rating for the given segment.
      - RATER_ID (optional) marks which rater did the rating. If not provided, RATING_NAME is used.
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
