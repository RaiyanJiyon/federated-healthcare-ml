# External Validation Plan: eICU-CRD Dataset Integration

## Executive Summary

**Reviewer Suggestion**: Single-dataset evaluation (MIMIC-IV only) is the biggest limitation. Add external validation on eICU-CRD to address generalizability concerns and substantially strengthen the paper.

**Strategic Impact**: 
- **HIGH LEVERAGE**: This is the single highest-priority change according to reviewers
- **Credibility**: Transforms the paper from "case study on one hospital system" to "validated methodology across independent health systems"
- **Publication Impact**: Dramatically increases acceptance probability for JBHI (IEEE Journal of Biomedical and Health Informatics)

---

## Phase 1: eICU-CRD Data Layer (Weeks 1-2)

### 1.1 Data Extraction from BigQuery

**Objective**: Extract eICU-CRD cohort matching MIMIC-IV structure and feature set.

**Rationale**: eICU-CRD is available in same BigQuery project as MIMIC-IV (physionet-data). You already have access. Subset of ~50 hospitals will provide independent validation without rerunning full dataset.

**Tasks**:

1. **Create BigQuery cohort query** (`src/data/eicu_loader.py`)
   - **Location**: Google BigQuery project `physionet-data`
   - **Key tables**: 
     - `physionet-data.eicu_crd.patient` (demographics, admission data)
     - `physionet-data.eicu_crd.vital` (vital signs, first 24h)
     - `physionet-data.eicu_crd.lab` (laboratory values, first 24h)
   - **Key differences from MIMIC-IV**:
     - eICU uses `patientunitstayid` (not stay_id)
     - Vital/lab data use `observationtime` (not charttime)
     - No direct ICU care unit labels; use `unittype` instead
     - Different itemid mapping (eICU-specific)

2. **Map eICU itemids to MIMIC features**
   - Create itemid lookup table:
     ```
     MIMIC Feature → MIMIC itemid → eICU itemid (if available)
     Example: heart_rate_mean → 220045 → 220045 (sometimes same, often different)
     ```
   - Handle missing features gracefully (some labs may not be in eICU)
   - Document any feature substitutions (e.g., alternative Lab IDs)

3. **Extract similar 31-feature set**
   - **Demographics** (4): age, gender_M, admission_type_emergency, insurance_primary
   - **Vitals** (13): heart_rate, sbp, mbp, resp_rate, temperature, spo2, glucose
   - **Labs** (12): creatinine, bun, sodium, potassium, bicarbonate, hemoglobin, wbc, platelets, lactate, bilirubin, inr, albumin
   - **Scores** (2): SOFA (computed or raw if available), simplified Charlson
   - **Target**: hospital_expire_flag or equivalent (`actualdischargestatus == 'Expired'`)

4. **Handle eICU-specific challenges**:
   - **Distributed data**: eICU spans 200+ hospitals (vs MIMIC-IV's single hospital). Use `hospitalid` for partitioning instead of ICU care unit
   - **Sparse labs**: Not all tests ordered at all hospitals. Handle missing labs via imputation/flagging
   - **Time alignment**: Use first 24h from admission to ICU (`unitadmittime`)

**Deliverable**: 
- `src/data/eicu_loader.py` (parallel to `loader.py` for MIMIC-IV)
- BigQuery SQL query cached to local CSV: `data/cache/eicu_cohort.csv` (~50K-100K rows)
- Feature mapping documentation: `src/data/eicu_feature_mapping.json`

---

### 1.2 Data Preprocessing & Cohort Validation

**Objective**: Create eICU cohort with identical preprocessing to MIMIC-IV for fair comparison.

**Tasks**:

1. **Extend preprocessing pipeline** (`src/data/preprocess.py`)
   - Add `dataset_name` parameter to preprocessing functions
   - Implement dataset-specific clinical clipping ranges (eICU uses different lab ranges):
     ```python
     CLINICAL_BOUNDS = {
         'mimic_iv': {
             'heart_rate_mean': (30, 200),
             'creatinine_max': (0, 10),
             ...
         },
         'eicu': {
             'heart_rate_mean': (20, 250),
             'creatinine_max': (0, 12),
             ...
         }
     }
     ```
   - Validate that preprocessing produces similar distributions across datasets

2. **Cohort filtering** (parallel to MIMIC-IV Phase 0)
   - Age ≥ 18, first ICU stay only
   - ICU LOS ≥ 4 hours
   - Complete mortality labels
   - Subset to ~50 hospital IDs (largest hospitals in eICU for stability)
   - Target cohort size: 50K-100K admissions

3. **Feature validation report**
   - Compare feature distributions (mean, std, quantiles) between MIMIC-IV and eICU
   - Identify dataset-specific gaps (e.g., if certain labs unavailable in eICU)
   - Document clinical plausibility

**Deliverable**:
- Updated `src/data/preprocess.py` with `dataset_name` parameter
- Cohort validation report: `results/eicu_cohort_validation.txt`
- Feature comparison plot: `results/plots/eicu_vs_mimic_feature_distributions.pdf`

---

### 1.3 Client Partitioning Strategy for eICU

**Objective**: Create non-IID federated clients from eICU data similar to MIMIC-IV care-unit partitioning.

**Challenge**: eICU data doesn't have "care units" like MIMIC-IV. Need alternative partitioning:

**Options** (choose one):
1. **Option A: Partition by hospital_id** (recommended)
   - Select top 7-10 hospitals by patient count
   - Simulate multi-hospital network (aligns with real-world federated scenario)
   - More realistic than MIMIC-IV (which is single hospital)

2. **Option B: Partition by unittype**
   - ICU types: SICU, MICU, CCU, NICU, etc. (similar to MIMIC-IV care units)
   - Less realistic (all within single hospital system)

**Recommendation**: Use **Option A (hospital-based partitioning)** because:
- More realistic: different hospitals → different patient populations, practices, EHR systems
- Stronger validation: if FL works across real hospitals, more generalizable
- Better matches actual deployment scenario

**Tasks**:

1. **Identify top hospitals by volume**
   - Query eICU for hospital_id distribution
   - Select 7-10 hospitals with ≥2000 patients each
   - Document hospital characteristics (academic vs community, region, etc.)

2. **Extend partitioning code** (`src/data/split.py`)
   - Add function `distribute_by_hospital()` (parallel to `distribute_by_care_unit()`)
   - Ensure minimum patient thresholds per hospital
   - Validate resulting non-IID distribution

3. **Compare heterogeneity metrics**
   - Calculate data heterogeneity ($\sigma^2_{het}$) for eICU hospital partition
   - Compare to MIMIC-IV care-unit partition
   - Report in paper: "eICU shows comparable/higher heterogeneity, validating generalization"

**Deliverable**:
- Updated `src/data/split.py` with hospital-based partitioning
- Hospital selection report: `results/eicu_hospital_selection.txt`
- Heterogeneity comparison: `results/eicu_mimic_heterogeneity_comparison.csv`

---

## Phase 2: Federated Learning Experiments on eICU (Weeks 2-3)

### 2.1 Unified Experiment Runner

**Objective**: Create dataset-agnostic experiment framework so we can run same experiments on both MIMIC-IV and eICU.

**Current state**: Experiments hardcoded for MIMIC-IV

**Tasks**:

1. **Refactor config system** (`src/config/config.py`)
   - Add `ACTIVE_DATASET` parameter: "mimic_iv" or "eicu_crd"
   - Create dataset-specific configuration sections:
     ```python
     DATASET_CONFIG = {
         'mimic_iv': {
             'cohort_cache': 'data/cache/mimic_iv_cohort.csv',
             'num_clients': 7,
             'client_partition_strategy': 'care_unit',
             'hospitals': None,
             'features_missing_ok': [],
         },
         'eicu_crd': {
             'cohort_cache': 'data/cache/eicu_cohort.csv',
             'num_clients': 7,  # top 7 hospitals
             'client_partition_strategy': 'hospital_id',
             'hospitals': ['00001', '00002', ...],  # top hospitals
             'features_missing_ok': ['albumin'],  # if not available
         }
     }
     ```

2. **Create data loader factory** (`src/data/__init__.py`)
   - Unified function `load_dataset(dataset_name)` that:
     - Loads correct cohort CSV (from cache)
     - Applies correct preprocessing
     - Returns (X, y, partition_info) regardless of dataset

3. **Update all experiments** to accept `--dataset` argument
   - Example: `python experiments/exp1_baseline.py --dataset mimic_iv --dataset eicu_crd`
   - Or: create new scripts like `exp1_baseline_eicu.py` (simpler approach for Phase 2)

**Deliverable**:
- Refactored `src/config/config.py` with dataset switching
- Updated `src/data/loader.py` to use loader factory
- Updated experiment scripts to accept dataset parameter

---

### 2.2 Core Experiments on eICU

**Objective**: Run same experiments as paper on eICU to validate generalization.

**Key experiments** (run these first for highest impact):

1. **Exp 1: Centralized Baseline**
   - Train centralized LR and XGBoost on eICU
   - **Expected**: Similar AUROC to MIMIC-IV (0.85-0.92 range)
   - **Success**: If eICU baseline ≥0.82 AUROC, dataset is appropriate

2. **Exp 2: Federated Baseline (FedAvg)**
   - Train federated LR on 7 eICU hospital clients
   - **Expected**: Federated AUROC within 1-2% of centralized
   - **Success**: Validates that FL works across independent hospitals (not just ICU departments)

3. **Exp 4: Aggregation Comparison** (optional, high compute)
   - Compare FedAvg vs FedProx vs FedF2 on eICU
   - **Expected**: FedF2 shows similar benefit as MIMIC-IV
   - **Success**: Confirms FedF2 generalizes

4. **Exp 8: Calibration** (critical for clinical trust)
   - Platt scaling, ECE, reliability curves
   - **Expected**: Calibration error similar to MIMIC-IV after Platt scaling
   - **Success**: Confirms calibration methodology works across datasets

**High-priority subset** (run first, takes ~2-4 hours):
- Exp 1 (centralized) on eICU
- Exp 2 (federated FedAvg) on eICU
- Exp 8 (calibration) on eICU

**Full validation** (if time/budget allows):
- All experiments above, plus FedProx, Byzantine robustness, DP-SGD

**Deliverable**:
- Results directory: `results/eicu/` (mirrors MIMIC-IV structure)
- Results tables: CSV files in `results/eicu/plots/`
- Summary report: `results/eicu_validation_summary.txt`

---

### 2.3 Multi-Dataset Results Comparison

**Objective**: Create unified comparison showing that FL approach generalizes across datasets.

**Tasks**:

1. **Create comparison table** 
   - Columns: Metric | MIMIC-IV | eICU-CRD | Difference
   - Rows: 
     - Centralized LR AUROC
     - Federated FedAvg AUROC
     - Federated AUROC loss (%)
     - Test Recall @ threshold
     - Expected Calibration Error (ECE)
     - Data heterogeneity ($\sigma^2_{het}$)
   - Example:
     ```
     Centralized LR AUROC    | 0.8914 | 0.8856 | 0.58%
     Federated FedAvg AUROC  | 0.8784 | 0.8723 | 0.70%
     AUROC Loss (%)          | 1.46%  | 1.50%  | -0.04pp
     Test Recall @ 0.39      | 85.2%  | 84.1%  | 1.1pp
     ECE (Platt-calibrated)  | 0.0091 | 0.0103 | -0.0012
     ```

2. **Statistical validation**
   - Compare AUROC with 95% CI across datasets
   - Perform equivalence test: Is eICU AUROC within 2% of MIMIC-IV?
   - Report p-values

3. **Qualitative analysis**
   - Discuss any dataset differences that emerge
   - Explain clinical reasons for AUROC differences
   - Document lessons learned about eICU data quality

**Deliverable**:
- Comparison table: `results/multi_dataset_comparison_table.csv`
- Statistical analysis: `results/multi_dataset_statistical_validation.txt`
- Visualization: `results/plots/multi_dataset_auroc_comparison.pdf`

---

## Phase 3: Paper Integration (Weeks 3-4)

### 3.1 Manuscript Revisions

**Objective**: Integrate eICU validation results into paper to address reviewer concern.

**Sections to revise**:

1. **Abstract** (add 1-2 sentences)
   - Mention eICU-CRD external validation
   - State validation result (e.g., "validated on eICU-CRD across 7 independent hospitals with comparable performance")

2. **Introduction** (add paragraph)
   - Acknowledge MIMIC-IV limitation explicitly
   - Justify eICU as external validation choice
   - Set expectation: "We further validate our approach on eICU-CRD..."

3. **Methodology > Distributed Patient Cohorts** (add subsection)
   - Section 3.2a: MIMIC-IV (existing)
   - Section 3.2b: NEW "eICU-CRD External Validation Cohort"
   - Describe hospital-based partitioning, feature mapping, cohort statistics

4. **Results** (add subsection)
   - NEW "4.4 External Validation on eICU-CRD"
   - Present comparison table (above)
   - Discuss generalization: "FedAvg achieves comparable AUROC on independent eICU hospitals, validating the approach generalizes beyond MIMIC-IV"
   - Address Byzantine robustness on eICU (if time permits)

5. **Discussion** (add paragraph)
   - Address generalizability: "Our approach validated on two independent datasets (MIMIC-IV single hospital, eICU-CRD multi-hospital) demonstrates robustness to dataset and institutional variation"
   - Discuss implications for real deployment

6. **Related Work** (update)
   - Mention eICU validation in healthcare FL literature context

**Manuscript word budget**:
- Current: ~8000 words (estimate)
- Addition: +500-800 words for eICU section
- New total: ~8500-8800 words (acceptable for JBHI)

**Deliverable**:
- Updated `paper/main.tex` with eICU sections
- Updated `paper/figures/` with eICU comparison plots
- Updated `paper/references.bib` if new citations needed

---

### 3.2 Table and Figure Generation

**Objective**: Create publication-quality figures comparing MIMIC-IV and eICU.

**New figures**:

1. **Figure X: Multi-Dataset AUROC Comparison**
   - Bar plot: Centralized and Federated AUROC for both datasets
   - Error bars: 95% CI
   - Reference line: 0.85 target AUROC
   - Caption: "FedAvg maintains comparable AUROC on eICU-CRD (7 hospital clients) vs MIMIC-IV..."

2. **Figure Y: Data Heterogeneity Comparison**
   - Histogram/violin plot: Feature distributions across clients
   - Comparison: MIMIC-IV care units vs eICU hospitals
   - Show that eICU exhibits comparable/higher heterogeneity

3. **Figure Z: Calibration Validation**
   - Reliability curves for both datasets (pre/post Platt)
   - ECE comparison bar plot
   - Demonstrates calibration methodology works universally

**New table**:

- **Table X: Multi-Dataset Experiment Results**
  - Comprehensive comparison (see Phase 2.3 above)
  - Include footnotes explaining any significant differences

**Deliverable**:
- LaTeX figures: `paper/figures/fig_eicu_auroc_comparison.pdf`
- LaTeX figures: `paper/figures/fig_heterogeneity_comparison.pdf`
- LaTeX figures: `paper/figures/fig_calibration_validation.pdf`
- Updated TeX table: embedded in `paper/main.tex`

---

### 3.3 Supplementary Material

**Objective**: Provide detailed eICU validation details without cluttering main paper.

**Supplementary sections**:

1. **eICU-CRD Data Processing Appendix**
   - Detailed itemid mappings
   - Feature validation plots for all 31 features
   - Cohort filtering flowchart
   - Hospital characteristics table

2. **eICU Experiment Details**
   - Hyperparameter settings (same as MIMIC-IV?)
   - Per-hospital results (not in main paper, too detailed)
   - Convergence plots per dataset

3. **Statistical Analysis**
   - Equivalence test details (specify equivalence margin)
   - Sensitivity analysis (what if we used different hospitals?)

**Deliverable**:
- `paper/appendix_eicu_validation.pdf` (standalone supplement)
- Or: supplementary material in IEEE submission format

---

## Phase 4: Deployment & Final Validation (Week 4)

### 4.1 Code Cleanup & Documentation

**Objective**: Make code reproducible and maintainable for reviewers.

**Tasks**:

1. **Add eICU data loading documentation**
   - Update README.md with eICU setup instructions
   - Document BigQuery access requirements
   - Add troubleshooting for eICU-specific issues

2. **Create reproducibility scripts**
   - Script to download eICU cohort from BigQuery
   - Script to run all eICU experiments in order
   - Script to generate comparison tables/figures

3. **Code cleanup**
   - Ensure no hardcoded paths (use config)
   - Add docstrings to new functions
   - Type hints for new functions

**Deliverable**:
- Updated README with eICU section
- `scripts/reproduce_eicu_validation.sh` (run all eICU experiments)
- `scripts/download_eicu_cohort.py` (BigQuery download helper)

---

### 4.2 Results Validation & Sanity Checks

**Objective**: Ensure eICU results are credible before submission.

**Sanity checks**:

1. **AUROC ranges**
   - Both datasets: AUROC > 0.82 (if not, investigate data quality)
   - AUROC loss < 5% in federated (if higher, may indicate model mismatch)

2. **Recall/Precision balance**
   - Similar recall at 0.39 threshold across datasets
   - If eICU recall is much higher/lower, may indicate mortality rate difference

3. **Convergence**
   - FedAvg converges in similar rounds across datasets
   - If eICU converges much slower, may indicate data heterogeneity

4. **Feature importance stability**
   - Top features consistent across datasets
   - If completely different, may indicate data quality issues

**Deliverable**:
- Sanity check report: `results/eicu_sanity_checks.txt`
- Flag any results requiring investigation/re-running

---

### 4.3 Submission Preparation

**Objective**: Package results for journal submission.

**Tasks**:

1. **Create submission artifacts**
   - Compiled PDF: `paper/main_eicu_validation.pdf`
   - All figures/tables in submission format
   - Supplementary material with eICU details

2. **Prepare reviewer response**
   - Draft response to single-dataset limitation:
     ```
     "We appreciate this feedback. The single-dataset limitation was significant.
     We have conducted external validation on eICU-CRD, extracting 7 hospital 
     systems covering ~80K admissions. Federated FedAvg achieves AUROC 0.8723 on 
     eICU (vs 0.8784 on MIMIC-IV), validating generalization across independent 
     hospital networks. Detailed results and feature mapping are in Section 3.2b 
     and Appendix A."
     ```

3. **Update all references**
   - BibTeX citations for eICU-CRD if not already there
   - Verify all new citations are formatted correctly

**Deliverable**:
- Final submission package with eICU validation integrated
- Draft reviewer response addressing single-dataset concern

---

## Timeline & Resource Estimate

### CPU-Only Path (Recommended - No GPU Needed)

| Phase | Task | Duration | Compute | Runtime |
|-------|------|----------|---------|---------|
| Phase 1.1 | BigQuery eICU extraction | 1-2 hours | Cloud (BigQuery) | ~5 min |
| Phase 1.2 | Preprocessing & validation | 1-2 hours | CPU | ~20-30 min |
| Phase 1.3 | Partitioning strategy | 1 hour | CPU | ~10 min |
| Phase 2.1 | Unified experiment framework | 2-3 hours | CPU | N/A (coding) |
| **Phase 2.2 Core** | **Exp 1,2,8 (LR only)** | **2-4 hours** | **CPU-only** | **~1-2 hours** |
| Phase 2.3 | Results comparison | 2-3 hours | CPU | ~30 min |
| Phase 3.1 | Manuscript revision | 3-5 hours | N/A | N/A (writing) |
| Phase 3.2 | Figure/table generation | 2-3 hours | CPU | ~30 min |
| Phase 3.3 | Supplementary material | 1-2 hours | N/A | N/A (writing) |
| Phase 4.1 | Code cleanup | 1-2 hours | CPU | N/A (coding) |
| Phase 4.2 | Validation & sanity checks | 1 hour | CPU | ~30 min |
| **TOTAL (MVP Path)** | | **~2-3 weeks** | **CPU-only** | **~2-3 hours runtime** |

**What this covers**:
- ✅ Centralized baseline on eICU
- ✅ Federated FedAvg on 7 hospitals
- ✅ Calibration validation
- ✅ Multi-dataset comparison
- ✅ Addresses reviewer's #1 concern

**GPU-accelerated path** (if ambitious): **3-4 weeks, +Byzantine/DP-SGD**, ~20 GPU hours

### Key Insight: CPU is Sufficient

For logistic regression and XGBoost on 50K-100K rows with 31 features:
- **No GPU needed** — CPU handles sklearn models efficiently
- **Runtime**: 1-2 hours for all core experiments
- **Bottleneck**: I/O (BigQuery) and preprocessing, not compute

---

## Success Criteria

### For Codebase:
- [ ] eICU cohort extracted and cached (50K-100K rows)
- [ ] Unified dataset-agnostic experiment framework
- [ ] All MIMIC-IV experiments runnable with `--dataset eicu_crd` flag
- [ ] No hardcoded paths or dataset assumptions

### For Results:
- [ ] Centralized LR AUROC on eICU: 0.82-0.90 (credible baseline)
- [ ] Federated FedAvg AUROC on eICU: ≥0.85 (meets clinical target)
- [ ] AUROC loss (centralized → federated) on eICU: <3% (validates FL)
- [ ] Multi-dataset AUROC comparison within 2% (validates generalization)
- [ ] ECE (Expected Calibration Error) < 0.02 after Platt scaling on both datasets

### For Manuscript:
- [ ] eICU validation section in main paper (500-800 words)
- [ ] Multi-dataset results table and figures
- [ ] Explicit statement of generalization: "validated on two independent datasets"
- [ ] Reviewer concern addressed in Introduction: "...we further validate on eICU-CRD..."

### For Reviewers:
- [ ] Addresses the single-dataset limitation head-on
- [ ] Demonstrates methodology is not MIMIC-IV-specific
- [ ] Provides comparable performance on independent hospital network
- [ ] Increases confidence in clinical utility

---

## Risk Mitigation

### Risk 1: eICU data quality issues
- **Mitigation**: Run Phase 1.2 cohort validation carefully. If centralized baseline AUROC <0.80, investigate missingness or feature distribution anomalies.
- **Fallback**: If eICU unsuitable, consider alternative: UK Biobank ICU subset, or HiRID dataset (though less convenient access)

### Risk 2: Federated performance degrades significantly on eICU
- **Mitigation**: Expected if eICU hospitals more heterogeneous. May indicate need for FedProx or FedF2. Investigate per-hospital feature drift.
- **Fallback**: Report result honestly ("...heterogeneous hospitals require robust aggregation...") and apply FedProx/FedF2 to show robustness.

### Risk 3: Compute budget exceeded
- **Mitigation**: Run minimal path first (Exp 1, 2, 8 only). Byzantine robustness and DP-SGD on eICU can be deferred or marked as future work.
- **Fallback**: Use smaller eICU subset (e.g., 5 hospitals) to reduce compute.

### Risk 4: BigQuery quotas or access issues
- **Mitigation**: You already have access in Google BigQuery. Verify BigQuery auth before starting.
- **Fallback**: If quota exceeded, use cached CSV (already downloaded) to avoid repeat queries.

---

## Integration with Paper Deadline

Assuming submission target: **Early June 2026**

- Week 1-2: Phase 1 (data + preprocessing)
- Week 2-3: Phase 2 (experiments)
- Week 3-4: Phase 3 (manuscript + figures)
- Week 4: Phase 4 (validation + submission)

**Expected submission**: End of Week 4 (late May / early June 2026)

---

## GPU Requirements & Alternatives

### Summary: Do You Need GPU?

**Short answer: NO. Not for the minimal viable path.**

### Breakdown by Phase

**Phase 1: Data Layer** → CPU-only
- BigQuery extraction (cloud-native, no local compute)
- Pandas preprocessing (CPU efficient)
- Runtime: ~30 min total

**Phase 2: Core Experiments** → CPU-only (logistic regression)
- Centralized LR: sklearn (CPU-efficient) → ~5-10 min
- Federated FedAvg: sklearn (CPU-efficient) → ~20-30 min
- Calibration: Platt scaling (trivial compute) → ~5-10 min
- **Total: 1-2 hours on CPU**

**Phase 2: Optional Advanced** → GPU helpful but not required
- XGBoost federated: Can run on CPU (slow, ~3-4 hours) or GPU (~30 min)
- Byzantine robustness: CPU-only (sequential) → ~2-3 hours
- DP-SGD experiments: Slow on CPU (~4-6 hours) or fast on GPU (~30 min)

### What This Means

**For addressing the reviewer's concern** (single-dataset limitation):
- ✅ Use **CPU-only, minimal path** (Exp 1, 2, 8 with LR)
- ✅ Runtime: 1-2 hours on your laptop
- ✅ No GPU needed
- ✅ Addresses the core criticism

**If you want to be comprehensive** (include Byzantine/DP-SGD):
- ⚠️ CPU runtime: 8-12 hours total (doable but slow)
- ✅ Or use free Google Colab GPU (2-3 hours instead)
- 🚀 Or use GCP Compute Engine (fast but costs $)

### Recommended: CPU-Only MVP Path

```bash
# On your local machine (no GPU needed)
# Total runtime: 1-2 hours

# Step 1: Extract eICU data from BigQuery
python src/data/eicu_loader.py  # 5 min

# Step 2: Preprocess
python -c "from src.data.preprocess import preprocess_eicu; preprocess_eicu()"  # 20 min

# Step 3: Run core experiments (CPU-only)
python experiments/exp1_baseline_eicu.py    # Centralized LR: 10 min
python experiments/exp2_noniid_eicu.py      # Federated FedAvg: 30 min
python experiments/exp8_calibration_eicu.py # Calibration: 10 min

# Step 4: Generate comparison table
python scripts/generate_comparison_table.py  # 5 min

# Total: ~1.5 hours, no GPU, all on your laptop ✅
```

### If You Want Faster Execution (Optional)

**Option 1: Google Colab (Free, ~2 GPU hours)**
- Upload eICU cohort CSV to Google Drive
- Run experiments on Colab free GPU
- Download results
- Time saved: ~6 hours CPU → 1 hour GPU

**Option 2: GCP Compute Engine ($2-5 per run)**
- Submit job to GCP Vertex AI
- Get results in parallel
- Most expensive but fastest

**Option 3: AWS SageMaker (Similar to GCP)**
- If you have AWS credits
- Similar timeline to GCP

**Recommendation**: **Start with local CPU**. If it takes 2 hours and you need results faster, then try Colab.

---

## Next Steps (Action Items)

### Immediate (This Week):
- [ ] Read this plan and confirm approach (hospital-based partitioning for eICU)
- [ ] Verify BigQuery access for eICU-CRD in your GCP project
- [ ] Estimate eICU cohort size (query sample: `SELECT COUNT(*) FROM eicu_crd.patient WHERE age >= 18`)

### Week 1:
- [ ] Implement Phase 1.1 (BigQuery cohort extraction)
- [ ] Implement Phase 1.2 (preprocessing with dataset parameter)
- [ ] Run cohort validation report

### Week 2:
- [ ] Implement Phase 1.3 (hospital-based partitioning)
- [ ] Implement Phase 2.1 (unified experiment framework)
- [ ] Run Phase 2.2 core experiments (Exp 1, 2, 8)

### Week 3:
- [ ] Run Phase 2.3 (results comparison)
- [ ] Revise manuscript (Phase 3.1)
- [ ] Generate figures (Phase 3.2)

### Week 4:
- [ ] Code cleanup (Phase 4.1)
- [ ] Final validation (Phase 4.2)
- [ ] Submit to journal

---

## Conclusion

This plan transforms your paper from single-dataset case study to **multi-dataset validated methodology**. The reviewer's suggestion to add eICU external validation is high-leverage and achievable within 3-4 weeks with proper planning.

**Key decision point**: Hospital-based partitioning for eICU (more realistic) vs. ICU-type partitioning (easier but less novel). Recommend hospital-based.

**Expected outcome**: Paper moves from "promising but limited to MIMIC-IV" to "validated across independent hospital networks, supporting real-world deployment."
