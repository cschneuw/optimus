# OPTIMUS: Predicting Multivariate Outcomes in Alzheimer’s Disease Using Multi-modal Data amidst Missing Values

## Authors
Christelle Schneuwly Diaz, Duy Thanh Vũ, Julien Bodelet, Duy-Cat Can, Guillaume Blanc,  
Haiteng Jiang, Lin Yao, Giuseppe Pantaleo and Oliver Y. Chén for ADNI

---

## Overview
**OPTIMUS** is a predictive and explainable machine-learning analysis for modeling multivariate cognitive and behavioral outcomes in Alzheimer’s Disease (AD) using multimodal data. It addresses challenges of missing data and aims to provide biological interpretability of predictive models.

The analysis covers:
- Missing data handling and imputation benchmarking
- Predictive modeling with classical and deep learning approaches
- Explainability using perturbation-based post-hoc model agnostic methods
- Brain mapping to contextualize feature attributions anatomically

---

## Objective
Alzheimer’s Disease involves complex neural, genetic, and proteomic factors affecting multiple cognitive domains. OPTIMUS aims to:
- Integrate multimodal data (neuroimaging, transcriptomics, genotyping, CSF biomarkers).
- Predict multivariate outcomes across memory, executive, language, and visuospatial domains.
- Handle missingness without losing biological signal.
- Identify key biomarkers and map them back to brain anatomy for interpretation.

---

## Methods

### Notebooks Overview
The project is organized into modular notebooks, each addressing a specific component of the OPTIMUS workflow:  

| Notebook                                  | Purpose                                                                                         |
|-------------------------------------------|-------------------------------------------------------------------------------------------------|
| **1-data_exploration.ipynb**              | Prepares and harmonizes multi-modal data (neuroimaging, genetic, proteomic), including feature engineering and scaling. |
| **2-imputation_analysis.ipynb**           | Benchmarks multiple imputation strategies (e.g., MissForest, KNN, IterativeImputer) for handling missing data across modalities. |
| **3-train-test_model_selection.ipynb**    | Implements predictive modeling with selected imputation strategies, using train-test split. Includes TabNet and classical models. |
| **4-loco-cv_model_selection.ipynb**       | Conducts leave-one-complete-out cross-validation (LOCO-CV) for model evaluation.               |
| **5-lomo-cv_model_selection.ipynb**       | Conducts leave-one-missing-out cross-validation (LOMO-CV) to assess model robustness to missingness. |
| **6-captum_feature_analysis.ipynb**       | Applies explainable AI methods (Captum perturbation-based explainers) to analyze feature importances per cognitive domain. |
| **7-feature_attribution_brain_maps.ipynb**| Projects neuroimaging feature importances onto cortical surfaces using Nilearn and the Schaefer 200-region atlas for functional interpretation. |
| **8-extra_hyperparameter_selection.ipynb**| Performs additional hyperparameter tuning for selected models on random patient subsets.         |

---

### Multi-modal Data
OPTIMUS is trained on data from **1,205 individuals** in the ADNI cohort:  
- **348** cognitively normal (CN)  
- **601** mild cognitive impairment (MCI)  
- **256** Alzheimer’s disease (AD)  

Modalities include:  
- **Neuroimaging** (MRI-derived cortical thickness and subcortical volumes)  
- **Genetic data** (APOE genotype, transcriptomics)  
- **Proteomics** (CSF biomarkers, plasma proteins)  

---

### Missing Data Handling
Imputation methods are benchmarked across multi-modal datasets using **2-imputation_analysis.ipynb**, assessing their ability to recover missing values while preserving inter-modal relationships.  

---

### Predictive Modeling
Predictive modeling pipelines are implemented across several notebooks:  
- **3-train-test_model_selection.ipynb:** Train-test split modeling.  
- **4-loco-cv_model_selection.ipynb:** Leave-one-complete-out CV.  
- **5-lomo-cv_model_selection.ipynb:** Leave-one-missing-out CV for robustness testing.  

Models include:  
- **Classical models:** MultiTaskElasticNet, MultiTaskLasso, Partial Least Squares (PLS), XGBoost  
- **Deep learning models:** TabNet (via PyTorch Tabular)  

The **8-extra_hyperparameter_selection.ipynb** notebook performs advanced hyperparameter searches.  

---

### Explainable AI (XAI)
The **6-captum_feature_analysis.ipynb** notebook integrates multiple perturbation-based explainability methods from Captum:  
- FeatureAblation  
- FeaturePermutation  
- Occlusion  
- ShapleyValueSampling  
- Lime  
- KernelShap  

Metrics such as **Infidelity** and **Sensitivity Max** are used for method evaluation.  

---

### Neuroimaging Visualization
The **7-feature_attribution_brain_maps.ipynb** notebook uses Nilearn to project cortical feature importances onto brain surfaces for functional interpretation, utilizing the Schaefer atlas.  

---

## Results
OPTIMUS identifies:  
- **Multi-modal biomarker patterns** relevant to specific cognitive domains.  
- Robust feature attributions across classical and deep learning models.  
- Neuroimaging regions whose importance is consistent across explainability methods.  

---

## Significance
This framework demonstrates:  
- The value of **multi-modal integration** in AD research.  
- Effective **missing data handling** strategies.  
- **Interpretable models** for biomarker discovery and hypothesis generation.  

---

## Data Availability
This study utilizes data from the **Alzheimer’s Disease Neuroimaging Initiative (ADNI)**.  
Access requires registration: [ADNI Data Access](https://adni.loni.usc.edu)  

---

## Code Availability
All notebooks and supporting scripts for preprocessing, imputation, modeling, and visualization will be made available at:  
🔗 [GitHub Repository](https://github.com/cschneuw/optimus)  
