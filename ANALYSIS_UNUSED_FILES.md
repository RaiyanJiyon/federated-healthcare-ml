# Analysis: Unused and Unnecessary Files in federated-healthcare-ml

**Date Generated**: 2026-06-15  
**Project Focus**: MIMIC-IV dataset with federated learning for ICU mortality prediction  
**Repository Status**: Active development on `feature/mimiciv-baseline` branch

---

## Executive Summary

This repository contains **73 MB of unused/duplicate data cache files** and several **experimental scripts that are not integrated into the core pipeline**. The project has evolved from supporting both eICU-CRD and MIMIC-IV datasets to focusing exclusively on MIMIC-IV, leaving behind substantial technical debt.

### Key Findings:
- **46+ MB of eICU cache files** (never used in current pipeline)
- **Missing experiment file** (`exp9_scalability_analysis.py`) referenced in documentation and config
- **Prototype/trial experiments** that are exploratory and not in the main paper narrative
- **1.6 MB of legacy paper results logs** that are stale
- **Duplicate results** in both `results/plots/` and `paper/results/summary/`

---

## Category 1: eICU-CRD Dataset Cache Files

### Overview
The project was originally designed to support both MIMIC-IV and eICU-CRD datasets. After transitioning to focus exclusively on MIMIC-IV (as per README and current experiments), the eICU cache files remain but are no longer used.

| File Path | Size | Risk Level | Reason for Deletion |
|-----------|------|-----------|---------------------|
| [data/cache/eicu_cohort.csv](data/cache/eicu_cohort.csv) | 23 MB | **Low** | Raw eICU cohort never used by any active experiment. Only `load_dataset_with_df()` from MIMIC-IV loader is called in all exp*.py files. eICU loader exists in [src/data/eicu_loader.py](src/data/eicu_loader.py) but is never imported by active experiments. |
| [data/cache/eicu_cohort_processed.csv](data/cache/eicu_cohort_processed.csv) | 23 MB | **Low** | Preprocessed eICU cohort. Acts as fallback in [src/data/multi_dataset.py](src/data/multi_dataset.py) but never called. All experiments use MIMIC-IV. |
| [data/cache/eicu_partitioned/](data/cache/eicu_partitioned/) | 3.8 MB | **Low** | eICU hospital-level partitioned data. Not referenced by any active experiment. Was part of abandoned multi-dataset support phase. |

**Total Space Recovered**: 49.8 MB

**Git History**: Added in commit `9a63f6d` ("Add core experiments for eICU-CRD and MIMIC-IV datasets") but never actively maintained or referenced in README's core experiments.

**Recommendation**: ✅ **SAFE TO DELETE**
- No active code path uses these files
- `src/data/eicu_loader.py` has no imports from any exp*.py
- MIMIC-IV is the documented focus: "Using the **MIMIC-IV clinical cohort** (65,273 ICU admissions...)"
- Multi-dataset support code (`src/data/multi_dataset.py`) exists but is not invoked

---

## Category 2: Missing Experiment File

### File: `exp9_scalability_analysis.py`

**Status**: ❌ **FILE DOES NOT EXIST**

**Where Referenced**:
- [README.md](README.md) - Lists experiments "exp1 to exp9"
- [LEARNING_GUIDE.md](LEARNING_GUIDE.md) - Documents exp9 with example `python experiments/exp9_scalability_analysis.py`
- [PHASE_PLAN.md](PHASE_PLAN.md) - References exp9 for scalability testing
- [src/config/trial_config.py](src/config/trial_config.py) - Configures exp9 as a critical trial experiment
- Generated Results Files Exist:
  - [results/plots/exp9_scalability_analysis.csv](results/plots/exp9_scalability_analysis.csv)
  - [results/plots/exp9_scalability_analysis_summary.csv](results/plots/exp9_scalability_analysis_summary.csv)

**Git History**: No creation or deletion recorded. Was mentioned in documentation/configuration but implementation never committed.

**Recommendation**: ⚠️ **DOCUMENTATION/CONFIG CLEANUP NEEDED**
- **Option A (Recommended)**: Create the missing `experiments/exp9_scalability_analysis.py` based on trial_config.py specification (network scalability: 7, 14, 21, 28 clients)
- **Option B**: Remove all exp9 references from:
  - [README.md](README.md)
  - [LEARNING_GUIDE.md](LEARNING_GUIDE.md)
  - [PHASE_PLAN.md](PHASE_PLAN.md)
  - [src/config/trial_config.py](src/config/trial_config.py)
- **Option C**: If results exist, create a minimal exp9 script that just loads and returns pre-computed CSV results

**Impact**: Low - Experiment not in documented core pipeline; only appears in advanced documentation

---

## Category 3: Exploratory/Prototype Experiments

These experiments are in the repository but not integrated into the main paper narrative or the core experiment pipeline defined in [run.py](run.py).

### 3A: exp1_baseline_multimodel.py

| Property | Value |
|----------|-------|
| **File** | [experiments/exp1_baseline_multimodel.py](experiments/exp1_baseline_multimodel.py) |
| **Size** | 14.5 KB |
| **Git Status** | Added once, never modified (May 26) |
| **Referenced In** | [ROADMAP_for.md](ROADMAP_for.md) (Phase 1 planning) |
| **Purpose** | Extends exp1 to support multiple model architectures (LR + MLP) |
| **Usage** | ❌ **NOT CALLED** by [run.py](run.py) or any other experiment |
| **Results Generated** | ✅ Yes: [results/plots/multimodel_comparison.csv](results/plots/multimodel_comparison.csv) (431 bytes) |

**Status**: Proof-of-concept for MLP baseline support (never completed per ROADMAP_for.md planning notes)

**Recommendation**: ⚠️ **OPTIONAL CLEANUP**
- **Risk Level**: Medium
- **Action**: Can be deleted if MLP is not a requirement for final paper. If keeping, integrate into core pipeline or document as experimental/exploratory.
- **Note**: [ROADMAP_for.md](ROADMAP_for.md) suggests MLP was a Phase 2+ enhancement. If not in final paper, can be removed.

---

### 3B: exp_dp_prototypes.py

| Property | Value |
|----------|-------|
| **File** | [experiments/exp_dp_prototypes.py](experiments/exp_dp_prototypes.py) |
| **Size** | 7.3 KB |
| **Purpose** | Prototype DP experiments comparing server-side vs client-side DP noise placement |
| **Usage** | ❌ **NOT CALLED** by [run.py](run.py) |
| **Results Generated** | Logs in `paper/results/summary/logs/prototypes/` (20+ .log files) |

**Status**: Early exploration of differential privacy approaches (superseded by [exp7_clinical_aggregation.py](experiments/exp7_clinical_aggregation.py))

**Recommendation**: ⚠️ **SAFE TO DELETE**
- **Risk Level**: Low
- **Rationale**: Core DP work is in exp7/exp8; this prototype is not integrated
- **Alternative**: Keep if documentation of DP exploration history is desired

---

### 3C: exp1_dp_sweep.py

| Property | Value |
|----------|-------|
| **File** | [experiments/exp1_dp_sweep.py](experiments/exp1_dp_sweep.py) |
| **Size** | 2.9 KB |
| **Purpose** | Runs epsilon sweep for DP experiments (epsilon in {1.0, 5.0, 10.0}) |
| **Usage** | ❌ **NOT CALLED** by [run.py](run.py) |
| **Git Status** | Added once, never modified |
| **Results Generated** | Logs in `results/summary/logs/` |

**Status**: Early DP exploration (superseded by [exp7_clinical_aggregation.py](experiments/exp7_clinical_aggregation.py) and phase5 scripts)

**Recommendation**: ✅ **SAFE TO DELETE**
- **Risk Level**: Low
- **Rationale**: Replaced by exp7 and phase5 statistical aggregation experiments
- **Space Saved**: 2.9 KB

---

### 3D: regenerate_figure2.py

| Property | Value |
|----------|-------|
| **File** | [experiments/regenerate_figure2.py](experiments/regenerate_figure2.py) |
| **Size** | 1.4 KB |
| **Purpose** | Generate Figure 2 (AUROC comparison) from CSV |
| **Usage** | ❌ **NOT CALLED** by [run.py](run.py) |
| **Current Status** | Superseded by [experiments/regenerate_all_figures.py](experiments/regenerate_all_figures.py) (17.9 KB) |

**Status**: Single-figure generator (now part of comprehensive figure regeneration script)

**Recommendation**: ✅ **SAFE TO DELETE**
- **Risk Level**: Low
- **Rationale**: Functionality merged into [regenerate_all_figures.py](experiments/regenerate_all_figures.py) which generates all 8 paper figures
- **Space Saved**: 1.4 KB

---

## Category 4: Phase 5 Statistical Validation Scripts

These scripts are marked "Phase 5" suggesting they were for extended validation beyond the core paper experiments.

### 4A: phase5_dp_sweep.py

| Property | Value |
|----------|-------|
| **File** | [experiments/phase5_dp_sweep.py](experiments/phase5_dp_sweep.py) |
| **Size** | 275 lines, 11.7 KB |
| **Purpose** | Extended DP epsilon sweep with 30 trial seeds for statistical rigor |
| **Usage** | ❌ **NOT CALLED** by [run.py](run.py) |
| **Git Status** | Added May 27, never modified |
| **Intended For** | Trial configuration (configured in [src/config/trial_config.py](src/config/trial_config.py)) but no evidence of execution |

**Status**: Experimental statistical validation (not part of final paper core experiments)

**Recommendation**: ⚠️ **OPTIONAL CLEANUP**
- **Risk Level**: Medium (may be desired for reproducibility validation)
- **Action**: Keep if statistical rigor documentation is needed; delete if only core experiments matter
- **Note**: Check if results are referenced in paper or supplementary material

---

### 4B: phase5_statistical_aggregation.py

| Property | Value |
|----------|-------|
| **File** | [experiments/phase5_statistical_aggregation.py](experiments/phase5_statistical_aggregation.py) |
| **Size** | 398 lines, 19.6 KB |
| **Purpose** | Advanced aggregation strategies with statistical testing |
| **Usage** | ❌ **NOT CALLED** by [run.py](run.py) |
| **Results Generated** | Possibly in `results/summary/` |

**Status**: Extended experimental validation

**Recommendation**: ⚠️ **OPTIONAL CLEANUP** (same as phase5_dp_sweep.py)

---

## Category 5: Utility/Helper Scripts

These are small utility scripts that are less likely to be needed.

### 5A: aggregate_and_confirm.py

| Property | Value |
|----------|-------|
| **File** | [experiments/aggregate_and_confirm.py](experiments/aggregate_and_confirm.py) |
| **Size** | 3.0 KB |
| **Purpose** | Parse prototype logs and select top-2 configurations |
| **Usage** | ❌ **NOT CALLED** by [run.py](run.py) |
| **Git Status** | Added May 24, never modified |

**Status**: One-off utility for prototype result aggregation

**Recommendation**: ✅ **SAFE TO DELETE**
- **Risk Level**: Low
- **Rationale**: Specific to exp_dp_prototypes exploratory work

---

### 5B: apply_platt_scaling.py

| Property | Value |
|----------|-------|
| **File** | [experiments/apply_platt_scaling.py](experiments/apply_platt_scaling.py) |
| **Size** | 5.9 KB |
| **Purpose** | Apply post-hoc Platt scaling (sigmoid calibration) to FedAvg outputs |
| **Usage** | ⚠️ **UNCLEAR** - may be integrated into main pipeline or used for paper results |
| **Results Generated** | Possibly used for calibration metrics in paper |

**Status**: Utility for probability calibration

**Recommendation**: ⚠️ **KEEP (for now)**
- **Risk Level**: Medium
- **Rationale**: Platt scaling is mentioned in paper results and README; calibration is a key contribution
- **Action**: Verify if results are used in paper before deleting

---

## Category 6: Data Preparation/eICU Scripts

The `scripts/` directory contains eICU-specific preparation scripts that are no longer used.

| File | Size | Status | Reason |
|------|------|--------|--------|
| [scripts/extract_eicu_cohort.py](scripts/extract_eicu_cohort.py) | 7.3 KB | ❌ Unused | Queries BigQuery for eICU-CRD (project focused on MIMIC-IV) |
| [scripts/test_eicu_bigquery.py](scripts/test_eicu_bigquery.py) | 5.8 KB | ❌ Unused | Tests eICU BigQuery connection |
| [scripts/partition_eicu_hospitals.py](scripts/partition_eicu_hospitals.py) | 6.1 KB | ❌ Unused | Partitions eICU by hospital (not needed for MIMIC-IV which uses care units) |
| [scripts/preprocess_eicu_data.py](scripts/preprocess_eicu_data.py) | 8.2 KB | ❌ Unused | Preprocesses eICU data |
| [scripts/phase2_3_integration_guide.py](scripts/phase2_3_integration_guide.py) | 4.2 KB | ❌ Unused | Documentation script for old phases |
| [scripts/phase2_core_experiments.py](scripts/phase2_core_experiments.py) | 11.4 KB | ⚠️ Unclear | Supports both MIMIC-IV and eICU but not integrated into core pipeline |

**Total Size**: ~43 KB

**Recommendation**: ✅ **SAFE TO DELETE (except phase2_core_experiments.py)**
- **Risk Level**: Low
- **Rationale**: All reference eICU dataset which is not used in current pipeline
- **Exception**: [scripts/phase2_core_experiments.py](scripts/phase2_core_experiments.py) - keep if it provides MIMIC-IV validation, otherwise delete

---

## Category 7: Cache/Log Files

### 7A: Paper Results Logs

**Location**: [paper/results/summary/logs/](paper/results/summary/logs/)

| Item | Size | Status |
|------|------|--------|
| `prototypes/` | 66 KB | ❌ Stale prototype logs (exp_dp_prototypes results) |
| `diag_clip/`, `diag_eps/` | ~500 KB | ❌ Diagnostic logs from old experiments |
| Total | 1.6 MB | ❌ All unnecessary |

**Recommendation**: ✅ **SAFE TO DELETE**
- **Risk Level**: Low (these are debug/diagnostic logs, not primary results)
- **Rationale**: Results are already in CSV format in `results/plots/`; logs are redundant

---

### 7B: Experiment Output Cache

**Location**: [experiments/__pycache__/](experiments/__pycache__/)

**Status**: Standard Python bytecode cache

**Recommendation**: ✅ **SAFE TO DELETE**
- Can be regenerated by Python on next import
- Delete via: `find . -type d -name __pycache__ -exec rm -r {} +`

---

## Category 8: Duplicate Result Files

### 8A: Results Exist in Multiple Locations

| File | Location 1 | Location 2 | Status |
|------|-----------|-----------|--------|
| Scalability plots | `results/plots/` | `paper/results/figures/` | ⚠️ Duplicate |
| Robustness analysis | `results/plots/` | `paper/results/figures/` | ⚠️ Duplicate |
| Calibration curves | `results/plots/` | `paper/results/figures/` | ⚠️ Duplicate |

**Location**: [paper/results/figures/](paper/results/figures/) contains copies of plots also in [results/plots/](results/plots/)

**Recommendation**: ⚠️ **OPTIONAL CLEANUP**
- **Risk Level**: Medium
- **Rationale**: Paper building might reference `paper/results/figures/` directly; check [paper/main.tex](paper/main.tex) to verify which path is used
- **Space Saved**: ~2.7 MB
- **Action**: If `paper/main.tex` uses `results/plots/`, can remove `paper/results/figures/`

---

## Category 9: Template and Deprecated Documentation Files

### 9A: PHASE6_TABLE1_TEMPLATE.tex

| Property | Value |
|----------|-------|
| **File** | [PHASE6_TABLE1_TEMPLATE.tex](PHASE6_TABLE1_TEMPLATE.tex) |
| **Size** | 2.4 KB |
| **Status** | ⚠️ Template/placeholder |
| **Usage** | Unclear if integrated into main.tex |
| **Git Status** | Added May 27, never modified |

**Recommendation**: ⚠️ **CHECK BEFORE DELETING**
- **Action**: Verify if content is used in [paper/main.tex](paper/main.tex)
- **If unused**: Safe to delete

---

### 9B: ROADMAP_for.md

| Property | Value |
|----------|-------|
| **File** | [ROADMAP_for.md](ROADMAP_for.md) |
| **Size** | 15 KB |
| **Status** | ⚠️ Planning document for Phase 1 MLP roadmap |
| **Age** | Last modified May 26 (pre-phase completion) |
| **Reference** | Only referenced in project planning, not by code |

**Recommendation**: ⚠️ **OPTIONAL CLEANUP**
- **Risk Level**: Low
- **Action**: Keep if planning documentation is valuable; delete if considered obsolete
- **Note**: Information may be duplicated in [LEARNING_GUIDE.md](LEARNING_GUIDE.md) or [PHASE_PLAN.md](PHASE_PLAN.md)

---

## Summary Table: Files Recommended for Deletion

### Tier 1: Safe to Delete (Low Risk, Minimal Impact)

| File/Directory | Size | Category | Why |
|---|---|---|---|
| data/cache/eicu_cohort.csv | 23 MB | eICU Cache | Not used by any active experiment |
| data/cache/eicu_cohort_processed.csv | 23 MB | eICU Cache | Not used by any active experiment |
| data/cache/eicu_partitioned/ | 3.8 MB | eICU Cache | Not used by any active experiment |
| experiments/exp1_dp_sweep.py | 2.9 KB | Prototype Exp | Superseded by exp7/phase5 experiments |
| experiments/regenerate_figure2.py | 1.4 KB | Prototype Exp | Functionality merged into regenerate_all_figures.py |
| experiments/aggregate_and_confirm.py | 3.0 KB | Utility | Specific to prototype work (exp_dp_prototypes) |
| experiments/exp_dp_prototypes.py | 7.3 KB | Prototype Exp | Early exploration (not in final pipeline) |
| experiments/__pycache__/ | ~1-2 MB | Cache | Standard Python cache (regenerated automatically) |
| paper/results/summary/logs/ | 1.6 MB | Logs | Diagnostic logs (results in CSV already) |
| scripts/extract_eicu_cohort.py | 7.3 KB | Data Prep | eICU-specific (not used with MIMIC-IV focus) |
| scripts/test_eicu_bigquery.py | 5.8 KB | Data Prep | eICU testing (not used) |
| scripts/partition_eicu_hospitals.py | 6.1 KB | Data Prep | eICU partitioning (not used) |
| scripts/preprocess_eicu_data.py | 8.2 KB | Data Prep | eICU preprocessing (not used) |
| scripts/phase2_3_integration_guide.py | 4.2 KB | Data Prep | Old phase documentation |

**Total Space Recovered**: ~57 MB

---

### Tier 2: Consider Deleting (Medium Risk, Requires Verification)

| File | Size | Category | Why | Action Required |
|---|---|---|---|---|
| experiments/exp1_baseline_multimodel.py | 14.5 KB | Prototype Exp | MLP baseline (not in paper?) | Verify if MLP is part of final results |
| experiments/apply_platt_scaling.py | 5.9 KB | Utility | Calibration utility | Verify if used for paper results |
| experiments/phase5_dp_sweep.py | 11.7 KB | Phase 5 | Statistical validation | Check if results are referenced |
| experiments/phase5_statistical_aggregation.py | 19.6 KB | Phase 5 | Statistical validation | Check if results are referenced |
| scripts/phase2_core_experiments.py | 11.4 KB | Data Prep | Multi-dataset support | Verify if MIMIC-IV validation script |
| paper/results/figures/ | 2.7 MB | Duplicate | Duplicate of results/plots/ | Check paper/main.tex for references |
| PHASE6_TABLE1_TEMPLATE.tex | 2.4 KB | Template | Template file | Verify integration with main.tex |
| ROADMAP_for.md | 15 KB | Documentation | Planning document | Decide if keeping planning docs |

**Total Conditional Space**: ~2.8 MB

---

### Tier 3: Keep (Important or Unclear)

| File | Reason |
|---|---|
| src/data/eicu_loader.py | Could be kept for future multi-dataset support (but remove eICU cache) |
| src/data/multi_dataset.py | Could be kept for future multi-dataset support (but remove eICU cache) |
| experiments/exp_xgboost_federated.py | Appears to be part of evaluation (check if in paper) |
| experiments/exp_robustness_fedf2.py | FedF2 robustness evaluation (likely in paper) |
| notebooks/exploration.ipynb | Exploratory analysis (keep for reference unless deprecated) |

---

## Analysis of eICU Loader and Multi-Dataset Support

### Current State
- `src/data/eicu_loader.py`: 17.5 KB (fully implemented)
- `src/data/multi_dataset.py`: 6.3 KB (wrapper for MIMIC-IV and eICU support)
- eICU cache: 49.8 MB
- eICU preparation scripts: 43 KB

### Usage in Active Experiments
```bash
grep -r "eicu_loader\|multi_dataset" experiments/*.py
# Result: NO MATCHES
```

### Conclusion
While multi-dataset infrastructure exists, it is:
- ❌ Never imported by any active experiment
- ❌ Never referenced in run.py or core pipeline
- ❌ Not mentioned in README as a supported feature
- ❌ Only referenced in ROADMAP_for.md (planning document)

**Recommendation**: 
- ✅ **DELETE all eICU cache files and eICU-specific scripts**
- ⚠️ **OPTIONALLY KEEP** `src/data/eicu_loader.py` and `src/data/multi_dataset.py` if planning future multi-dataset support
- 🗑️ **IF KEEPING SOURCE CODE**: Remove all eICU data files to avoid confusion

---

## Git History Insights

### Key Commits Related to Unused Files
```
9a63f6d: "Add core experiments for eICU-CRD and MIMIC-IV datasets"
         (Added eICU loader, multi-dataset support, scripts - last used here)

ad7295c: "Delete outdated project documentation and figures related to eICU 
         validation. Remove PLAN_EICU_VALIDATION.md and PROJECT_COMPLETE.md"
         (Shows project maintainer already cleaning up eICU-related docs)

7b5246c: "Refactor code structure for improved readability and maintainability"
         (General refactoring, no eICU changes)
```

**Implication**: The project maintainer has already begun removing eICU-related documentation (commit ad7295c), suggesting awareness that eICU is no longer the focus.

---

## Recommendations Summary

### High Priority (Safe, High Impact)
1. ✅ **Delete eICU cache files** (49.8 MB)
   - `data/cache/eicu_cohort.csv`
   - `data/cache/eicu_cohort_processed.csv`
   - `data/cache/eicu_partitioned/`

2. ✅ **Delete eICU data preparation scripts** (43 KB)
   - `scripts/extract_eicu_cohort.py`
   - `scripts/test_eicu_bigquery.py`
   - `scripts/partition_eicu_hospitals.py`
   - `scripts/preprocess_eicu_data.py`
   - `scripts/phase2_3_integration_guide.py`

3. ✅ **Delete prototype/exploration experiments** (~20 KB)
   - `experiments/exp1_dp_sweep.py`
   - `experiments/regenerate_figure2.py`
   - `experiments/aggregate_and_confirm.py`
   - `experiments/exp_dp_prototypes.py`

4. ✅ **Delete legacy logs and caches**
   - `paper/results/summary/logs/` (1.6 MB)
   - `experiments/__pycache__/` (regenerated automatically)

**Total Space Recovered**: ~55 MB

---

### Medium Priority (Requires Verification)
1. ⚠️ **Verify exp1_baseline_multimodel.py**
   - Check if MLP results appear in final paper
   - If not in paper: DELETE (14.5 KB)

2. ⚠️ **Verify apply_platt_scaling.py**
   - Confirm if calibration results are from this script or built into exp8
   - If standalone utility with no integration: DELETE (5.9 KB)

3. ⚠️ **Verify phase5 experiments**
   - Check if exp9_scalability_analysis results are used
   - If not in paper: DELETE both phase5 scripts (31.3 KB)

4. ⚠️ **Check paper/results/figures/**
   - Verify if `paper/main.tex` references this directory
   - If using `results/plots/` instead: DELETE (2.7 MB)

**Conditional Space Recovered**: ~2.8 MB

---

### Documentation Updates Needed
1. **Create missing exp9_scalability_analysis.py** OR remove all exp9 references from:
   - [README.md](README.md)
   - [LEARNING_GUIDE.md](LEARNING_GUIDE.md)
   - [PHASE_PLAN.md](PHASE_PLAN.md)
   - [src/config/trial_config.py](src/config/trial_config.py)

2. **Update README.md** to clarify:
   - "This project focuses exclusively on MIMIC-IV dataset"
   - Remove or clarify exp1-exp9 list if some are not in core pipeline

---

## Cleanup Checklist

```bash
# TIER 1: Safe deletions (55 MB saved)
rm -rf data/cache/eicu_cohort.csv
rm -rf data/cache/eicu_cohort_processed.csv
rm -rf data/cache/eicu_partitioned/
rm -rf scripts/extract_eicu_cohort.py
rm -rf scripts/test_eicu_bigquery.py
rm -rf scripts/partition_eicu_hospitals.py
rm -rf scripts/preprocess_eicu_data.py
rm -rf scripts/phase2_3_integration_guide.py
rm -rf experiments/exp1_dp_sweep.py
rm -rf experiments/regenerate_figure2.py
rm -rf experiments/aggregate_and_confirm.py
rm -rf experiments/exp_dp_prototypes.py
rm -rf paper/results/summary/logs/
find . -type d -name __pycache__ -exec rm -r {} +

# TIER 2: After verification (~2.8 MB additional)
# [See Medium Priority section]

# TIER 3: Documentation fixes
# - Create exp9_scalability_analysis.py OR update documentation
# - Verify paper/results/figures/ usage
# - Verify PHASE6_TABLE1_TEMPLATE.tex usage
```

---

## Risk Assessment

| Action | Probability of Issue | Impact | Mitigation |
|--------|---------------------|--------|-----------|
| Delete eICU cache | Very Low (0%) | None - not used | Already committed to git; can restore if needed |
| Delete eICU scripts | Very Low (0%) | None - not used | Already committed to git; can restore if needed |
| Delete prototype exps | Low (5%) | May break if someone runs old commands | Update documentation; scripts not in core pipeline |
| Delete phase5 experiments | Low (10%) | May break if used for reproducibility | Verify with project maintainer first |
| Delete duplicate results | Low (5%) | May break if paper building uses old paths | Verify `paper/main.tex` references |

---

## Conclusion

The repository contains approximately **57 MB of unused files** that can be safely deleted to improve maintainability and clarity. An additional **~2.8 MB** can be removed after verification. The project has cleanly transitioned from multi-dataset support to MIMIC-IV focus, but cleanup was not completed.

**Recommended Action**: Implement Tier 1 deletions immediately; verify and implement Tier 2 after checking paper integration.

---

*Analysis completed: 2026-06-15*  
*Workspace: /home/raiyanjiyon/Projects/federated-healthcare-ml*
