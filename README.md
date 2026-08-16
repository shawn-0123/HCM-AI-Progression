# HCM-AI-Progression
Achieved top 8% placement among ~350 teams in the international open-entry data science competition: CardI-HACK Data Challenge

Competition Host: IHU ICAN
- IHU ICAN is a leading French translational research institute dedicated to cardiometabolic diseases, combining clinical expertise and scientific innovation to drive advances in patient care.

# Overview
Hypertrophic cardiomyopathy (HCM) is the most common inherited cardiac disease and a major cause of arrhythmias, heart failure, and sudden cardiac death. While clinical imaging and biomarkers are routinely used to assess disease severity, genetic risk stratification remains challenging, as single pathogenic variants fail to explain the wide heterogeneity in patient outcomes.

This project develops a machine learning framework that integrates clinical data, pathogenic variants, and common genetic polymorphisms (SNPs) to predict patient prognosis.

# Repository layout

The pipeline is split across three notebooks and one shared module. Run them in order.

| File | Contents |
| --- | --- |
| `cardi_utils.py` | Shared, target-agnostic code: data loading, the two competition metrics (weighted log-loss with baseline rescaling; weighted quadratic kappa with ordinal thresholds), LightGBM cross-validation plumbing, and SHAP-based SNP ranking. |
| `01_severity_model.ipynb` | **OUTCOME SEVERITY** (binary). Ranks SNPs by weighted mean \|SHAP\|, builds features (top-20 SNPs raw + PCA components + SNP sum), random-searches LightGBM parameters under 5-fold stratified CV, and predicts on test. Writes `pred_severity.csv`. |
| `02_mace_model.ipynb` | **OUTCOME MACE** (ordinal, 3 classes). Same feature construction on the full clinical panel, with class probabilities collapsed to an ordinal score and two cut points tuned per fold to maximise weighted QWK. Writes `pred_mace.csv`. |
| `03_submission.ipynb` | Merges the two prediction files on `trustii_id` into the final submission. |

Both model notebooks expect `cardihack_final_train.csv` and `cardihack_final_test.csv` in the working directory; the competition data is not redistributed here.

Requires `pandas`, `numpy`, `lightgbm`, and `scikit-learn`.
