# `data/`

This directory holds the artefacts the pipeline reads at run time. None
of the agent trajectories are checked in — the two upstream benchmarks
have their own licences, so we ship the loaders and ask you to fetch the
sources yourself.

## What you should populate before running

| Path | Provided by | How to obtain |
| --- | --- | --- |
| `data/ASSEBench/` | [`Eligius-MS/ASSEBench`](https://github.com/Eligius-MS/ASSEBench) | `git clone https://github.com/Eligius-MS/ASSEBench data/ASSEBench` |
| `data/R-Judge/`   | [`Lordog/R-Judge`](https://github.com/Lordog/R-Judge) | `git clone https://github.com/Lordog/R-Judge data/R-Judge` |
| `data/master_table.json` | this repo | run `python scripts/prepare_data.py` |
| `data/rewrites_all.json` | this repo | run `python scripts/generate_rewrites.py` |

## What the loader expects

The loaders in `pij/data_loader.py` read:

```
data/ASSEBench/ASSEBench/dataset/AgentJudge-strict.json
data/ASSEBench/ASSEBench/dataset/AgentJudge-loose.json
data/R-Judge/data/<category>/<file>.json
```

If your local layout differs, pass an explicit `root=` argument to
`load_assebench` / `load_rjudge`.
