# AlphaFold 3 + GNINA Virtual Screening Pipeline State

## Overview & Status
- **Date / Time**: 2026-07-21 23:24 CEST
- **Cluster Host**: ETH Euler HPC (`eu-login-27`)
- **Python Environment**: Python 3.13.0 (`venv/` using `stack/2025-06`)
- **Pipeline Execution Status**: **STAGE 1 CPU MSAs COMPLETE** | **STAGE 2 AF3 GPU PREDICTIONS COMPLETE** | **GNINA RESCORING & RANKING COMPLETE**

---

## 1. Stage 1: MSA Generation (CPU) — COMPLETED [OK]
- **Master Job ID**: `8034011` (Elapsed: 36 min 51 sec)
- **Alignments**: All 4 target TFs (`acrr`, `arac`, `cysb`, `tyrr`) generated in `alphafold3_msas/`.
- **Indexing**: `.af3io_data_index.json` created for `af3io` MSA reuse.

---

## 2. Stage 2: AF3 GPU Structure Predictions — COMPLETED [OK]
- **Predictions Directory**: [alphafold3_predictions/](file:///cluster/project/beltrao/lucla/repos/TF_met_StrucSpec/alphafold3_predictions) (20 ZIP files)
- **Fixes Applied**:
  1. Generic `--gpus=1` broadened cluster queue access.
  2. `singularity_args: '--env XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"'` for Compute Capability 7.x GPUs (Quadro RTX 6000, RTX 2080 Ti).
  3. `run_alphafold_args: '--flash_attention_implementation=xla'` for cross-GPU architecture compatibility.
  4. Lowercase directory post-processing zip fix in `workflow/targets/alphafold3_datafill_predictions.smk`.

---

## 3. Downstream: GNINA Redocking & Candidate Ranking — COMPLETED [OK]
- **GNINA Rescoring Job ID**: `8058872` (Mode: `redock`, Autobox Expansion: `8.0 Å`)
- **Output Scores CSV**: [data/processed/gnina_scores.csv](file:///cluster/project/beltrao/lucla/repos/TF_met_StrucSpec/data/processed/gnina_scores.csv) (20 pairs scored)
- **Ranked Candidates Report**: [results/ranked_pairings_report.csv](file:///cluster/project/beltrao/lucla/repos/TF_met_StrucSpec/results/ranked_pairings_report.csv)

### Diagnostic Subset Top Ranking Summary
| Target TF | Top Ranked Candidate (Consensus AF3+GNINA) | AF3 Rank | Consensus Rank | Consensus Score |
|---|---|---|---|---|
| **AraC** | **D-fucose** | #1 | #1 | **2.8911** |
| **AcrR** | **proflavin** | #1 | #1 | **2.2991** |
| **CysB** | **Thiosulphate** | #1 | #1 | **1.2848** |
| **TyrR** | **L-phenylalanine** | #1 | #1 | **1.7722** |

*All 4 target transcription factors successfully identified their true positive effector ligands as the #1 top-ranked candidates!*
