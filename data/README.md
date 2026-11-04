# `data/`

This directory holds the artefacts the pipeline reads at run time. None
of the agent trajectories are checked in: the upstream benchmarks have
their own licences, so we ship the loaders and ask you to fetch the
sources yourself. The certified rewrite set and the annotation records
used in the paper are included for reproducibility (see below).

## What you should populate before running

| Path | Provided by | How to obtain |
| --- | --- | --- |
| `data/ASSEBench/` | [`Eligius-MS/ASSEBench`](https://github.com/Eligius-MS/ASSEBench) | `git clone https://github.com/Eligius-MS/ASSEBench data/ASSEBench` |
| `data/R-Judge/`   | [`Lordog/R-Judge`](https://github.com/Lordog/R-Judge) | `git clone https://github.com/Lordog/R-Judge data/R-Judge` |
| `data/master_table.json` | this repo | run `python scripts/prepare_data.py` |
| `data/rewrites_all.json` | this repo (shipped) | already included; regenerate with `python scripts/generate_rewrites.py` only for a new policy |

## What the loader expects

The loaders in `pij/data_loader.py` read:

```
data/ASSEBench/ASSEBench/dataset/AgentJudge-strict.json
data/ASSEBench/ASSEBench/dataset/AgentJudge-loose.json
data/R-Judge/data/<category>/<file>.json
```

If your local layout differs, pass an explicit `root=` argument to
`load_assebench` / `load_rjudge`.

## Certification materials shipped with the paper

| Path | Content |
| --- | --- |
| `rewrites_all.json` | Final rewrite set used in the main analysis: 500 items, transforms T1 to T5. |
| `raw_rewrites_all.json` | First-pass rewrites. The 45 T1/T2/T4 pairs that failed unanimous certification were regenerated. |
| `annotations/certified-equivalence-annotator-{A,B,C}.json` | First-round six-dimension ratings of every T1/T2/T4 pair by each annotator. |
| `annotations/certified-equivalence-annotator-{A,B,C}-round2.json` | Ratings of the regenerated pairs. |
| `annotations/near-equivalence-sample-annotator-{A,B,C}.json` | Ratings of the T3/T5 sample used for the perturbation-success rate. |
| `annotations/annotation_ui.py` | Annotation interface and six-dimension codebook. |
| `verify_certified_equivalence.py` | Checks the complete annotation, regeneration, and unanimous-certification chain. |

Provenance: rewrites were generated from the paper's base policy by
GPT-5.4-thinking (temperature 0.3) and rated by three independent,
anonymized annotators on six dimensions (deontic force, policy scope,
exception set, burden of proof, default decision rule, implied risk
threshold). A pair is certified only when all three annotators rate all
six dimensions as preserved. Run the check with:

```bash
python data/verify_certified_equivalence.py
```
