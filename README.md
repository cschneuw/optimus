# OPTIMUS: Predicting Multivariate Outcomes in Alzheimer’s Disease Using Multi-modal Data amidst Missing Values

## Authors
Christelle Schneuwly Diaz, Duy Thanh Vũ, Julien Bodelet, Duy-Cat Can, Guillaume Blanc,  
Haiting Jiang, Lin Yao, Holger Auner, Gilles Allali, Giuseppe Pantaleo, ADNI, Oliver Y. Chén  

---

## Overview
**OPTIMUS** is a predictive, modular, and explainable machine-learning framework designed to model **multivariate cognitive and behavioral outcomes** in Alzheimer’s Disease (AD) using **multi-modal data**, while explicitly addressing **missing values**. OPTIMUS integrates:

- **Missing data handling** strategies to recover missing values while preserving biological relevance.
- **Attention-based predictive modeling** tailored for structured multi-modal data.
- **Explainable AI (XAI)** to uncover interpretable relationships between biomarkers and cognitive outcomes.

---

## Objective
Alzheimer’s Disease is driven by a complex interplay of neural, genetic, and proteomic factors, impacting multiple cognitive domains. Traditional approaches often predict **univariate outcomes**, such as disease progression or severity scores. OPTIMUS goes beyond by:

- **Integrating multi-modal data**, including imaging, genetic, and proteomic features.
- **Predicting multivariate outcomes**, capturing multiple cognitive and behavioral functions simultaneously.
- **Handling missing data** with dedicated strategies to maximize usable data.

---

## Methods

### 1. Multi-modal Data
OPTIMUS is trained on data from **1,205 individuals**, spanning:
- **346** cognitively normal (CN) participants
- **608** individuals with mild cognitive impairment (MCI)
- **251** individuals diagnosed with Alzheimer’s disease (AD)

**Data modalities include:**
- **Neuroimaging** (e.g., MRI-based features)
- **Genetic data** (e.g., gene expression and APOE genotype)
- **Proteomic profiles** (e.g., CSF biomarkers)

---

### 2. Missing Data Handling
OPTIMUS applies **generalized imputation techniques** rather than strictly modality-specific imputation, ensuring that missing values are addressed while maintaining data integrity.

---

### 3. Predictive Modeling
OPTIMUS predicts **multivariate cognitive outcomes**, including:
- Executive function
- Language
- Memory
- Visuospatial function

**Core modeling approach:**
- **TabNet**, a deep learning model optimized for tabular data, enhanced with:
  - **Sparse feature selection** via attention mechanisms.
  - **Comprehensive evaluation** across multiple validation strategies.

**Validation strategies include:**
- **Train-test split** for initial evaluation.
- **Leave-one-complete-sample-out cross-validation** to test generalizability.
- **Leave-one-missing-sample-out cross-validation** to assess robustness in missing data scenarios.

---

### 4. Explainable AI (XAI)
OPTIMUS incorporates explainability directly into its modeling pipeline, using TabNet’s attention masks to:
- Identify the **most influential features per cognitive outcome**.
- Aggregate feature importance across decision steps.
- Compare attention-based explanations with alternative techniques, including:
  - **Permutation importance**
  - **Shapley values**

**Key visual outputs include:**
- **Volcano plots** highlighting differentially expressed genes.
- **Heatmaps** summarizing normalized gene expression profiles.
- **Aggregated feature importance plots** across cognitive domains.

---

## Results
- OPTIMUS identifies **multi-modal biomarkers** spanning neuroimaging, genetic, and proteomic data.
- These biomarkers **jointly but differentially** predict distinct cognitive outcomes.
- The model uncovers **many-to-many predictive pathways**, linking biomarkers to specific cognitive functions.

---

## Significance
OPTIMUS demonstrates the power of:
- **Multi-modal data integration** to capture the heterogeneous pathology of AD.
- **Imputation strategies** tailored to preserve data integrity.
- **Attention-based models** to enhance prediction accuracy and interpretability.
- **Explainable AI** to reveal biologically meaningful relationships between biomarkers and cognitive decline.

This work advances our understanding of Alzheimer’s Disease by offering a robust, interpretable framework for modeling cognitive decline.

---

## Data Availability
This study utilizes data from the **Alzheimer’s Disease Neuroimaging Initiative (ADNI)**.  
Access requires registration and approval via: [ADNI Data Access](https://adni.loni.usc.edu)

---

## Code Availability
All code for preprocessing, imputation, modeling, and visualization will be made available at:  
🔗 [GitHub Repository (placeholder link)](https://github.com/your-repo-link-here)

---

## Citation
If you use OPTIMUS in your work, please cite:

