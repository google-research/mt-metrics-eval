# MT Metrics Eval

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
cd mt_metrics_eval
pip install .
```

## Running from the command line

Start by downloading the database (this is also required before using the API):

```bash
alias mtme='python3 -m mt_metrics_eval.mtme'
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

Evaluate metric score files containing tab-separated "system-name score"
entries. For system-level correlations, supply one score per system. For
document-level or segment-level correlations, supply one score per document or
segment, grouped by system, in the same order as text generated using `--echo`
(the same order as the WMT test-set file). Granularity is determined
automatically:

```bash
examples=$HOME/.mt-metrics-eval/mt-metrics-eval/wmt20/metric-scores/en-de

mtme -t wmt20 -l en-de < $examples/sentBLEU.sys.score
mtme -t wmt20 -l en-de < $examples/sentBLEU.doc.score
mtme -t wmt20 -l en-de < $examples/sentBLEU.seg.score
```

Compare to MQM gold scores instead of WMT gold scores:

```bash
mtme -t wmt20 -l en-de -g mqm < $examples/sentBLEU.sys.score
```

Generate correlations for two metrics files, and perform tests to determine
whether they are significantly different:

```bash
mtme -t wmt20 -l en-de -i $examples/sentBLEU.sys.score -c $examples/COMET.sys.score
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

WMT correlations for a new metric, all granularities:

```python
from mt_metrics_eval import data

def MyMetric(out, ref):
  """Return a scalar score for given output/reference texts."""
  return -sum(abs(len(a) - len(b)) for a, b in zip(out, ref))

evs = data.EvalSet('wmt20', 'en-de')
sys_scores, doc_scores, seg_scores = {}, {}, {}
ref = evs.ref
for s, out in evs.sys_outputs.items():
  sys_scores[s] = [MyMetric(out, ref)]
  doc_scores[s] = [MyMetric(out[b:e], ref[b:e]) for b, e in evs.docs.values()]
  seg_scores[s] = [MyMetric([o], [r]) for o, r in zip(out, ref)]

# Official WMT correlations.
print('sys Pearson:', evs.Correlation('sys', sys_scores).Pearson()[0])
print('doc Kendall:', evs.Correlation('doc', doc_scores).KendallLike()[0])
print('seg Kendall:', evs.Correlation('seg', seg_scores).KendallLike()[0])
```

Correlations using alternative gold scores:

```python
from mt_metrics_eval import data

def MyMetric(out, ref):
  """Return a scalar score for given output/reference texts."""
  return -sum(abs(len(a) - len(b)) for a, b in zip(out, ref))

evs = data.EvalSet('wmt20', 'en-de')
scores = {s: [MyMetric(out, evs.ref)] for s, out in evs.sys_outputs.items()}

print('MQM Pearson:', evs.Correlation('sys', scores, 'mqm').Pearson()[0])
print('pSQM Pearson:', evs.Correlation('sys', scores, 'psqm').Pearson()[0])
print('cSQM Pearson:', evs.Correlation('sys', scores, 'csqm').Pearson()[0])

# The correlations above aren't comparable to WMT, since the *qm gold scores are
# available only for a subset of system outputs. To get comparable WMT scores,
# we need to explicitly specify these systems.
qm_sys = set(evs.Scores('sys', 'mqm')) - evs.human_sys_names
print('WMT Pearson:', evs.Correlation('sys', scores, sys_names=qm_sys).Pearson()[0])
```

New correlations for a stored metric:

```python
from mt_metrics_eval import data

# Eval set will load more slowly, since we're reading in all stored metrics.
evs = data.EvalSet('wmt20', 'en-de', read_stored_metric_scores=True)
scores = evs.Scores('sys', scorer='BLEURT-extended')

qm_sys = set(evs.Scores('sys', 'mqm')) - evs.human_sys_names
qm_sys_bp = qm_sys | {'Human-B.0', 'Human-P.0'}

wmt = evs.Correlation('sys', scores, sys_names=qm_sys)
mqm = evs.Correlation('sys', scores, gold_scorer='mqm', sys_names=qm_sys)
wmt_bp = evs.Correlation('sys', scores, sys_names=qm_sys_bp)
mqm_bp = evs.Correlation('sys', scores, gold_scorer='mqm', sys_names=qm_sys_bp)

print('BLEURT-ext WMT Pearson for qm systems:', wmt.Pearson()[0])
print('BLEURT-ext MQM Pearson for qm systems:', mqm.Pearson()[0])
print('BLEURT-ext WMT Pearson for qm systems plus human:', wmt_bp.Pearson()[0])
print('BLEURT-ext MQM Pearson for qm systems plus human:', mqm_bp.Pearson()[0])
```

## Credits

Inspired by and loosely modeled on
[SacreBLEU](https://github.com/mjpost/sacrebleu).

This is not an official Google product.
